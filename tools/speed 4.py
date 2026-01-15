import traceback
from turtle import width
#处理单张图像，分步骤
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import pywt
import os
import cv2
import heartpy as hp
from matplotlib.font_manager import FontProperties
from numpy import convolve
from scipy import signal
from scipy.interpolate import CubicSpline, make_interp_spline, interp1d, interpolate
from scipy.ndimage import binary_erosion, binary_dilation, binary_fill_holes, binary_closing, uniform_filter1d
from scipy.signal import savgol_filter, find_peaks, hilbert, butter, filtfilt
import pytesseract
from datetime import datetime
import re
import matplotlib.patheffects as pe
import neurokit2 as nk
from scipy.spatial import ConvexHull
from skimage.measure import find_contours

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False
large_font = FontProperties(size=40, weight='bold')  # 30pt 加粗

class DopplerEnvelopeAnalyzer:
    def __init__(self, prf=17, target_band=(0.8, 4.0), spectrum_region_ratio=(0.85, 1.0),
                 baseline_exclude_height=20, text_region_width_ratio=0.1,
                 left_text_width_ratio=0.18, left_text_height_ratio=0.7,
                 speed_scale_height_ratio=(0.3, 0.9), speed_scale_width_ratio=0.15):
        """初始化多普勒频谱分析器"""
        self.prf = prf
        self.target_band = target_band
        self.spectrum_region_ratio = spectrum_region_ratio
        self.wavelet = 'db6'
        self.level = 6
        self.baseline_exclude_height = baseline_exclude_height
        self.text_region_width_ratio = text_region_width_ratio
        self.left_text_width_ratio = left_text_width_ratio
        self.left_text_height_ratio = left_text_height_ratio
        self.speed_scale_height_ratio = speed_scale_height_ratio
        self.speed_scale_width_ratio = speed_scale_width_ratio
        self.speed_values = []  # 存储速度刻度值
        self.speed_positions = []  # 存储速度刻度对应的位置
        self.max_speed = None  # 最高速度值
        self.min_speed = None  # 最低速度值
        self.max_speed_pos = None  # 最高速度对应的像素位置
        self.min_speed_pos = None  # 最低速度对应的像素位置

    def load_image(self, image_path, return_original=False):
        """加载图像，返回处理用（掩膜后）灰度图和原始RGB图"""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"图像文件不存在: {image_path}")

            img_original = Image.open(image_path).convert("RGB")
            width, height = img_original.size

            # 掩膜左侧文字区域
            img_process = img_original.copy()
            draw = ImageDraw.Draw(img_process)
            text_area_width = int(width * self.left_text_width_ratio)
            text_area_height = int(height * self.left_text_height_ratio)
            draw.rectangle([(0, 0), (text_area_width, text_area_height)], fill=(0, 0, 0))

            img_gray = np.array(img_process.convert('L'), dtype=np.float32)

            if return_original:
                return img_gray, np.array(img_original)
            else:
                return img_gray

        except Exception as e:
            print(f"图像加载错误: {str(e)}")
            return (None, None) if return_original else None

    def detect_right_text_region(self, img_gray):
        """检测右侧文字区域并返回起始列"""
        h, w = img_gray.shape
        text_region_start_col = int(w * (1 - self.text_region_width_ratio))
        text_region = img_gray[:, text_region_start_col:]
        sobel_x = cv2.Sobel(text_region, cv2.CV_64F, 1, 0, ksize=3)
        sobel_x = np.abs(sobel_x)
        sobel_x = np.mean(sobel_x, axis=0)
        text_threshold = np.mean(sobel_x) + 1.0 * np.std(sobel_x)
        text_cols = np.where(sobel_x > text_threshold)[0]
        if len(text_cols) > 0:
            text_start_col = text_region_start_col + text_cols[0]
            return text_start_col
        return w

    def locate_spectrum(self, img_gray):
        """定位多普勒频谱区域"""
        try:
            h, w = img_gray.shape
            text_start_col = self.detect_right_text_region(img_gray)
            if text_start_col < w - 20:
                w = text_start_col

            region_start = int(h * self.spectrum_region_ratio[0])
            region_end = int(h * self.spectrum_region_ratio[1])
            roi_candidate = img_gray[region_start:region_end, :w]

            baseline_height = min(self.baseline_exclude_height, roi_candidate.shape[0] // 5)
            if baseline_height > 0:
                roi_candidate = roi_candidate[:-baseline_height, :]
                region_end -= baseline_height

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            enhanced = clahe.apply(roi_candidate.astype(np.uint8))
            _, binary = cv2.threshold(enhanced, 30, 255, cv2.THRESH_BINARY)

            horizontal_sum = np.sum(binary, axis=0)
            non_zero_cols = np.where(horizontal_sum > 5)[0]
            if len(non_zero_cols) < 10:
                raise ValueError("有效信号列不足，使用默认区域")

            consecutive_groups = []
            current_group = [non_zero_cols[0]]
            for col in non_zero_cols[1:]:
                if col == current_group[-1] + 1:
                    current_group.append(col)
                else:
                    consecutive_groups.append(current_group)
                    current_group = [col]
            consecutive_groups.append(current_group)
            longest_group = max(consecutive_groups, key=len)
            if len(longest_group) < 15:
                raise ValueError("有效信号段过短，使用默认区域")

            x_start = longest_group[0]
            x_end = longest_group[-1]
            return x_start, region_start, x_end - x_start + 1, region_end - region_start

        except Exception as e:
            print(f"频谱定位错误: {str(e)}，使用默认区域")
            h, w = img_gray.shape
            text_start_col = self.detect_right_text_region(img_gray) if h > 0 else w
            w = min(w, text_start_col)
            y_start = int(h * self.spectrum_region_ratio[0])
            y_height = int(h * 0.15)
            return 0, y_start, w, y_height

    def extract_envelopes(self, spectrum_roi,smooth_method='moving_avg'):
        """改进的包络线提取方法 - 多种平滑方法"""
        try:
            h, w = spectrum_roi.shape
            upper_envelope = np.full(w, np.nan)
            lower_envelope = np.full(w, np.nan)
            valid_width = int(w * 0.995)
            if valid_width <= 0:
                return upper_envelope, lower_envelope # 宽度无效时直接返回空包络
            total_groups = 0

            # 1. 预处理
            filtered_roi = cv2.GaussianBlur(spectrum_roi.astype(np.float32), (5, 5), 1.0)
            plt.figure(figsize=(18, 6))  # 扩展画布宽度以容纳新增步骤
            #
            # # 2. 滤波后图像
            plt.subplot(2, 3, 1)
            plt.imshow(filtered_roi, cmap='gray')
            plt.text(0.01, 1.02, "A", transform=plt.gca().transAxes,
                     fontproperties=large_font, color='black')  # 大字号字母
            plt.text(0.05, 1.02, " Gaussian Filtering", transform=plt.gca().transAxes,
                     fontsize=22, color='black')  # 正常字号文本
            plt.axis('off')

            # 3. 动态阈值计算与二值化
            valid_roi = filtered_roi[:, :valid_width]
            global_median = np.median(valid_roi)
            global_std = np.std(valid_roi)
            global_max = np.max(valid_roi)
            dynamic_threshold = min(global_median + 0.5 * global_std, 0.1 * global_max)
            thresholded = np.zeros_like(filtered_roi)
            thresholded[filtered_roi > dynamic_threshold] = 255
            plt.subplot(2, 3, 2)
            plt.imshow(thresholded, cmap='gray')
            plt.text(0.01, 1.02, "B", transform=plt.gca().transAxes,
                     fontproperties=large_font, color='black')  # 大字号字母
            plt.text(0.05, 1.02, " Dynamic Threshold Binarization", transform=plt.gca().transAxes,
                     fontsize=22, color='black')  # 正常字号文本
            plt.axis('off')

            # 4. 候选点分组（颜色区分）
            candidate_points = np.zeros_like(filtered_roi)
            group_visualization = np.zeros((h, w, 3), dtype=np.uint8)  # 彩色分组图
            # 新增：存储原始包络点（未插值前）
            raw_upper_points = np.full(w, np.nan)
            raw_lower_points = np.full(w, np.nan)

            for x in range(valid_width):
                column = filtered_roi[:, x]
                candidate_pixels = np.where(column > dynamic_threshold)[0]
                #备用阈值
                if len(candidate_pixels) < 1:
                    secondary_threshold = max(global_median + 0.2 * global_std, 20)
                    candidate_pixels = np.where(column > secondary_threshold)[0]
                    if len(candidate_pixels) < 1:
                        continue

                # 密度聚类分组
                if len(candidate_pixels) > 1:
                    density_threshold = 15
                    groups = []
                    current_group = [candidate_pixels[0]]
                    for i in range(1, len(candidate_pixels)):
                        if candidate_pixels[i] - candidate_pixels[i - 1] <= density_threshold:
                            current_group.append(candidate_pixels[i])
                        else:
                            groups.append(current_group)
                            current_group = [candidate_pixels[i]]
                    groups.append(current_group)
                    total_groups += len(groups)
                    # 分组可视化（随机颜色）
                    for i, group in enumerate(groups):
                        color = (np.random.randint(100, 255),
                                 np.random.randint(100, 255),
                                 np.random.randint(100, 255))
                        for y in group:
                            group_visualization[y, x] = color

                    # 选择最优组并记录原始包络点（关键中间步骤）
                    best_group = max(groups, key=lambda g: (np.ptp(g), len(g)))
                    upper_pixel = np.min(best_group)
                    lower_pixel = np.max(best_group)
                    raw_upper_points[x] = upper_pixel  # 原始上包络点（未插值）
                    raw_lower_points[x] = lower_pixel  # 原始下包络点（未插值）
                    upper_envelope[x] = upper_pixel
                    lower_envelope[x] = lower_pixel

            # 4. 候选点分组可视化
            plt.subplot(2, 3, 3)
            plt.imshow(group_visualization)
            plt.text(0.01, 1.02, "C", transform=plt.gca().transAxes,
                     fontproperties=large_font, color='black')  # 大字号字母
            plt.text(0.05, 1.02, " Candidate Point Grouping", transform=plt.gca().transAxes,
                     fontsize=22, color='black')  # 正常字号文本
            plt.axis('off')
            # # 新增步骤：原始包络点可视化 (候选点 -> [原始包络点] -> 插值包络)
            plt.subplot(2, 3, 4)
            plt.imshow(filtered_roi, cmap='gray')
            # 绘制原始上包络点 (红色圆点) 和下包络点 (蓝色圆点)
            plt.scatter(np.where(~np.isnan(raw_upper_points))[0],
                        raw_upper_points[~np.isnan(raw_upper_points)],
                        c='red', s=3)
            plt.scatter(np.where(~np.isnan(raw_lower_points))[0],
                        raw_lower_points[~np.isnan(raw_lower_points)],
                        c='blue', s=3)
            plt.text(0.01, 1.02, "D", transform=plt.gca().transAxes,
                     fontproperties=large_font, color='black')  # 大字号字母
            plt.text(0.05, 1.02, " Optimal Group(Envelope Point)", transform=plt.gca().transAxes,
                     fontsize=22, color='black')  # 正常字号文本
            # plt.title(r"$\fontsize{30pt}{36pt}\selectfont\textbf{D}$" + " Optimal group (envelope)", fontsize=20, loc='left')
            plt.axis('off')
            #
            #
            plt.subplot(2, 3, 5)
            plt.imshow(filtered_roi, cmap='gray')
            plt.plot(raw_upper_points, 'r-', linewidth=1)
            plt.plot(raw_lower_points, 'b-', linewidth=1)
            plt.text(0.01, 1.02, "E", transform=plt.gca().transAxes,
                     fontproperties=large_font, color='black')  # 大字号字母
            plt.text(0.05, 1.02, " Optimal Group(Envelope)", transform=plt.gca().transAxes,
                     fontsize=22, color='black')  # 正常字号文本
            # plt.title(r"$\fontsize{30pt}{36pt}\selectfont\textbf{E}$" + " Optimal group (envelope)", fontsize=20, loc='left')  # Modified
            plt.axis('off')

            # 6. 直接使用原始包络点
            upper_envelope = raw_upper_points
            lower_envelope = raw_lower_points


            # 7. 平滑处理（最终步骤）
            # from scipy.signal import savgol_filter
            # window_size = min(15, valid_width // 4)
            # if window_size % 2 == 0:
            #     window_size += 1
            # # 上包络平滑
            # valid_mask_upper = ~np.isnan(upper_envelope[:valid_width])
            # if np.sum(valid_mask_upper) > window_size:
            #     valid_values = upper_envelope[:valid_width][valid_mask_upper].reshape(-1)
            #     smoothed_values = savgol_filter(valid_values, window_size, 2, mode='interp')
            #     upper_envelope[:valid_width][valid_mask_upper] = smoothed_values
            #     upper_envelope[:valid_width] = cv2.medianBlur(
            #         upper_envelope[:valid_width].astype(np.float32).reshape(1, -1), 3).flatten()
            # # 下包络平滑
            # valid_mask_lower = ~np.isnan(lower_envelope[:valid_width])
            # if np.sum(valid_mask_lower) > window_size:
            #     valid_values = lower_envelope[:valid_width][valid_mask_lower].reshape(-1)
            #     smoothed_values = savgol_filter(valid_values, window_size, 2, mode='interp')
            #     lower_envelope[:valid_width][valid_mask_lower] = smoothed_values
            #     lower_envelope[:valid_width] = cv2.medianBlur(
            #         lower_envelope[:valid_width].astype(np.float32).reshape(1, -1), 3).flatten()
            # upper_envelope[valid_width:] = np.nan
            # lower_envelope[valid_width:] = np.nan


            def smooth_envelope(envelope, method, window_size=None):
                """内部平滑函数"""
                valid_mask = ~np.isnan(envelope[:valid_width])
                valid_values = envelope[:valid_width][valid_mask]

                if len(valid_values) < 3:
                    return envelope

                if window_size is None:
                    window_size = min(15, valid_width // 4)
                    if window_size % 2 == 0:
                        window_size += 1  # 确保奇数窗口

                if method == 'savgol':
                    # 原始Savitzky-Golay滤波+中值滤波
                    smoothed = savgol_filter(
                        valid_values,
                        window_length=window_size,
                        polyorder=2,
                        mode='interp'
                    )
                    smoothed = cv2.medianBlur(
                        smoothed.astype(np.float32).reshape(1, -1),
                        3
                    ).flatten()

                elif method == 'moving_avg':
                    # 移动平均
                    weights = np.ones(window_size) / window_size
                    smoothed = np.convolve(valid_values, weights, mode='same')
                    # 边界处理
                    half_window = window_size // 2
                    smoothed[:half_window] = valid_values[:half_window]
                    smoothed[-half_window:] = valid_values[-half_window:]

                elif method == 'gaussian':
                    # 高斯滤波
                    kernel_size = min(15, len(valid_values))
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    smoothed = cv2.GaussianBlur(
                        valid_values.astype(np.float32).reshape(1, -1),
                        (kernel_size, 1), 1.0
                    ).flatten()

                elif method == 'lowess':
                    # 局部加权回归
                    from statsmodels.nonparametric.smoothers_lowess import lowess
                    x = np.where(valid_mask)[0]
                    smoothed = lowess(valid_values, x, frac=0.1, return_sorted=False)

                elif method == 'spline':
                    # 样条平滑
                    from scipy.interpolate import UnivariateSpline
                    x = np.where(valid_mask)[0]
                    spl = UnivariateSpline(x, valid_values, s=len(valid_values) * np.var(valid_values) * 0.1)
                    smoothed = spl(x)

                elif method == 'kalman':
                    # 卡尔曼滤波
                    from pykalman import KalmanFilter
                    kf = KalmanFilter(
                        transition_matrices=[1],
                        observation_matrices=[1],
                        initial_state_mean=valid_values[0],
                        initial_state_covariance=1,
                        observation_covariance=1.0
                    )
                    smoothed, _ = kf.smooth(valid_values)
                    smoothed = smoothed.flatten()

                else:
                    raise ValueError(f"未知的平滑方法: {method}")

                # 将平滑后的值放回原数组
                result = envelope.copy()
                result[:valid_width][valid_mask] = smoothed
                return result

            upper_envelope = smooth_envelope(upper_envelope, smooth_method)
            lower_envelope = smooth_envelope(lower_envelope, smooth_method)

            # 最终平滑包络可视化
            plt.subplot(2, 3, 6)
            plt.imshow(filtered_roi, cmap='gray')
            plt.plot(upper_envelope, 'r-', linewidth=1.5)
            plt.plot(lower_envelope, 'b-', linewidth=1.5)
            plt.text(0.01, 1.02, "F", transform=plt.gca().transAxes,
                     fontproperties=large_font, color='black')  # 大字号字母
            plt.text(0.05, 1.02, " Smoothed Envelope", transform=plt.gca().transAxes,
                     fontsize=22, color='black')  # 正常字号文本
            plt.tight_layout()
            plt.axis('off')
            savepath=r'D:\桌面文件\1.png'
            plt.savefig(savepath, bbox_inches='tight', dpi=300)
            plt.show()


            print(f"候选点分组总数: {total_groups}")
            return upper_envelope, lower_envelope

        except Exception as e:
            print(f"包络提取错误: {str(e)}")
            return np.full(w, np.nan), np.full(w, np.nan)
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    from scipy.signal import hilbert, convolve
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    from scipy.signal import butter, filtfilt
    from scipy.ndimage import uniform_filter1d

    # def extract_envelopes(self, spectrum_roi):
    #     """基于低通滤波的包络提取（带平滑对比可视化）"""
    #     try:
    #         h, w = spectrum_roi.shape
    #         upper_envelope = np.full(w, np.nan)
    #         lower_envelope = np.full(w, np.nan)
    #
    #         # -------------------------- 1. 区域裁剪与预处理 -------------------------
    #         valid_start = int(w * 0.01)
    #         valid_end = int(w * 0.99)
    #         valid_width = valid_end - valid_start
    #         if valid_width <= 0 or h < 50:
    #             return upper_envelope, lower_envelope
    #
    #         # 裁剪并灰度化（确保单通道处理）
    #         spectrum_cropped = cv2.cvtColor(spectrum_roi, cv2.COLOR_BGR2GRAY)[:, valid_start:valid_end] if len(
    #             spectrum_roi.shape) == 3 else spectrum_roi[:, valid_start:valid_end]
    #         spectrum_cropped = spectrum_cropped.astype(np.float32)
    #
    #         # 垂直方向平滑（抑制高频噪声，保留频谱主体）
    #         filtered_roi = uniform_filter1d(spectrum_cropped, size=5, axis=0)  # 垂直方向低通滤波
    #
    #         # -------------------------- 2. 低通滤波生成原始包络（核心） -------------------------
    #         def lowpass_envelope(column, cutoff_freq=0.1):
    #             """对单列信号进行低通滤波，提取包络主体"""
    #             # 设计Butterworth低通滤波器（保留低频成分）
    #             b, a = butter(N=2, Wn=cutoff_freq, btype='low', analog=False)
    #             # 零相位滤波（避免包络偏移）
    #             lowpass_signal = filtfilt(b, a, column)
    #             return lowpass_signal
    #
    #         # 提取每列的上下包络（基于低通滤波结果）
    #         upper_raw = np.zeros(valid_width)
    #         lower_raw = np.zeros(valid_width)
    #         for x in range(valid_width):
    #             column = filtered_roi[:, x]
    #             # 低通滤波提取低频主体
    #             lowpass_col = lowpass_envelope(column, cutoff_freq=0.15)  # 截止频率0.15（经验值）
    #
    #             # 动态阈值：取低通信号最大值的40%作为边界阈值
    #             threshold = np.max(lowpass_col) * 0.4
    #             if threshold < 10:  # 避免噪声导致阈值过低
    #                 threshold = 10
    #
    #             # 上包络：从顶部向下找第一个超过阈值的点
    #             upper = np.argmax(lowpass_col > threshold)  # 第一个超过阈值的索引
    #             # 下包络：从底部向上找第一个超过阈值的点（反转数组后取argmax）
    #             lower = len(lowpass_col) - 1 - np.argmax(lowpass_col[::-1] > threshold)
    #
    #             upper_raw[x] = upper
    #             lower_raw[x] = lower
    #
    #         # -------------------------- 3. 移动平均平滑（方法不变） -------------------------
    #         def moving_average_smooth(data, window_size=11):
    #             """移动平均平滑：加权平均抑制高频噪声"""
    #             if window_size % 2 == 0:
    #                 window_size += 1  # 确保窗口为奇数
    #             # 卷积实现移动平均（边界处理保留原始值）
    #             weights = np.ones(window_size) / window_size
    #             smoothed = np.convolve(data, weights, mode='same')
    #             # 修复边界效应（前半窗口和后半窗口保留原始值）
    #             half_window = window_size // 2
    #             smoothed[:half_window] = data[:half_window]
    #             smoothed[-half_window:] = data[-half_window:]
    #             return smoothed
    #
    #         # 自适应窗口大小（根据频谱宽度动态调整，避免过平滑）
    #         window_size = min(15, valid_width // 8)  # 窗口范围：5~15
    #         upper_smooth = moving_average_smooth(upper_raw, window_size)
    #         lower_smooth = moving_average_smooth(lower_raw, window_size)
    #
    #         # -------------------------- 4. 可视化对比（平滑前 vs 平滑后） -------------------------
    #         plt.figure(figsize=(16, 6))  # 宽屏布局，左右对比
    #
    #         # 左图：低通滤波提取的原始包络（含高频抖动）
    #         plt.subplot(1, 2, 1)
    #         plt.imshow(filtered_roi, cmap='gray')
    #         plt.plot(upper_raw, 'r-', linewidth=1.5)
    #         plt.plot(lower_raw, 'b-', linewidth=1.5)
    #         plt.title("低通滤波提取的原始包络", fontsize=12)
    #         plt.legend(loc='upper right')
    #         plt.axis('off')
    #
    #         # 右图：移动平均平滑后的包络（抑制噪声）
    #         plt.subplot(1, 2, 2)
    #         plt.imshow(filtered_roi, cmap='gray')
    #         plt.plot(upper_smooth, 'r-', linewidth=1.5)
    #         plt.plot(lower_smooth, 'b-', linewidth=1.5)
    #         plt.title(f"移动平均平滑（窗口大小={window_size}）", fontsize=12)
    #         plt.legend(loc='upper right')
    #         plt.axis('off')
    #
    #         plt.tight_layout()
    #         plt.show()
    #
    #         # -------------------------- 5. 结果映射回原图 -------------------------
    #         upper_envelope[valid_start:valid_end] = upper_smooth
    #         lower_envelope[valid_start:valid_end] = lower_smooth
    #         # 边界限制（确保包络在图像范围内）
    #         upper_envelope = np.clip(upper_envelope, 0, h - 1)
    #         lower_envelope = np.clip(lower_envelope, 0, h - 1)
    #         # 无效区域标记为NaN
    #         upper_envelope[:valid_start] = np.nan
    #         upper_envelope[valid_end:] = np.nan
    #         lower_envelope[:valid_start] = np.nan
    #         lower_envelope[valid_end:] = np.nan
    #
    #         return upper_envelope.astype(np.float32), lower_envelope.astype(np.float32)
    #
    #     except Exception as e:
    #         print(f"包络提取错误: {str(e)}")
    #         return np.full(w, np.nan), np.full(w, np.nan)
    # def extract_envelopes(self, spectrum_roi):
    #     """基于希尔伯特变换的包络提取（带平滑对比可视化）"""
    #     try:
    #         h, w = spectrum_roi.shape
    #         upper_envelope = np.full(w, np.nan)
    #         lower_envelope = np.full(w, np.nan)
    #
    #         # 1. 区域裁剪（保留有效区域）
    #         valid_start = int(w * 0.01)
    #         valid_end = int(w * 0.99)
    #         valid_width = valid_end - valid_start
    #         if valid_width <= 0 or h < 50:
    #             return upper_envelope, lower_envelope
    #
    #         # 2. 预处理（灰度化+垂直方向平滑）
    #         spectrum_cropped = cv2.cvtColor(spectrum_roi, cv2.COLOR_BGR2GRAY)[:, valid_start:valid_end] if len(
    #             spectrum_roi.shape) == 3 else spectrum_roi[:, valid_start:valid_end]
    #         filtered_roi = cv2.blur(spectrum_cropped, ksize=(1, 5)).astype(np.float32)
    #
    #         # 3. 希尔伯特变换包络提取（核心修改部分）
    #         def hilbert_envelope(column):
    #             """对单列信号进行希尔伯特包络提取"""
    #             analytic_signal = hilbert(column)
    #             return np.abs(analytic_signal)
    #
    #         # 提取每列的上下包络
    #         upper_raw = np.zeros(valid_width)
    #         lower_raw = np.zeros(valid_width)
    #         for x in range(valid_width):
    #             column = filtered_roi[:, x]
    #             env = hilbert_envelope(column)
    #
    #             # 上包络：找到第一个超过最大能量50%的点（从顶部向下）
    #             max_energy = np.max(env)
    #             upper = np.argmax(env > 0.5 * max_energy)
    #
    #             # 下包络：找到最后一个超过最大能量50%的点（从底部向上）
    #             lower = len(env) - 1 - np.argmax(env[::-1] > 0.5 * max_energy)
    #
    #             upper_raw[x] = upper
    #             lower_raw[x] = lower
    #
    #         # 4. 移动平均平滑（保持不变）
    #         def moving_average(data, window=11):
    #             if window % 2 == 0: window += 1
    #             return convolve(data, np.ones(window) / window, mode='same')
    #
    #         window_size = min(15, valid_width // 10)
    #         upper_smooth = moving_average(upper_raw, window_size)
    #         lower_smooth = moving_average(lower_raw, window_size)
    #
    #         # 5. 可视化对比（平滑前 vs 平滑后）
    #         plt.figure(figsize=(16, 6))
    #         # 左图：希尔伯特提取的原始包络
    #         plt.subplot(1, 2, 1)
    #         plt.imshow(filtered_roi, cmap='gray')
    #         plt.plot(upper_raw, 'r-', linewidth=1.5)
    #         plt.plot(lower_raw, 'b-', linewidth=1.5)
    #         plt.title("希尔伯特变换提取的原始包络", fontsize=12)
    #         plt.legend()
    #         plt.axis('off')
    #
    #         # 右图：平滑后的包络
    #         plt.subplot(1, 2, 2)
    #         plt.imshow(filtered_roi, cmap='gray')
    #         plt.plot(upper_smooth, 'r-', linewidth=1.5)
    #         plt.plot(lower_smooth, 'b-', linewidth=1.5)
    #         plt.title(f"移动平均平滑 (窗口={window_size})", fontsize=12)
    #         plt.legend()
    #         plt.axis('off')
    #
    #         plt.tight_layout()
    #         plt.show()
    #
    #         plt.figure(figsize=(10, 6))  # 独立图形1
    #         plt.imshow(filtered_roi, cmap='gray')
    #         plt.plot(upper_raw, 'r-', linewidth=1.5)
    #         plt.plot(lower_raw, 'b-', linewidth=1.5)
    #         plt.axis('off')
    #         plt.tight_layout()
    #         # 保存到桌面（文件名带时间戳，避免覆盖）
    #         save_path1 = r'D:\桌面文件\1.png'
    #         plt.savefig(save_path1, bbox_inches='tight', dpi=300)
    #         plt.show()  # 显示独立窗口
    #
    #         # 图2：平滑后包络（独立窗口显示并保存）
    #         plt.figure(figsize=(10, 6))  # 独立图形2
    #         plt.imshow(filtered_roi, cmap='gray')
    #         plt.plot(upper_smooth, 'r-', linewidth=1.5)
    #         plt.plot(lower_smooth, 'b-', linewidth=1.5)
    #         plt.axis('off')
    #         plt.tight_layout()
    #         # 保存到桌面
    #         save_path2 = r'D:\桌面文件\2.png'
    #         plt.savefig(save_path2, bbox_inches='tight', dpi=300)
    #         plt.show()  # 显示独立窗口
    #         # 映射回原图坐标
    #         upper_envelope[valid_start:valid_end] = upper_smooth
    #         lower_envelope[valid_start:valid_end] = lower_smooth
    #         return np.clip(upper_envelope, 0, h - 1), np.clip(lower_envelope, 0, h - 1)
    #
    #     except Exception as e:
    #         print(f"包络提取错误: {str(e)}")
    #         return np.full(w, np.nan), np.full(w, np.nan)

    def extract_speed_scale_region(self, img_rgb):
        """提取速度刻度区域并识别速度值"""
        try:
            height, width, _ = img_rgb.shape
            scale_width = int(width * self.speed_scale_width_ratio)
            scale_left_col = width - scale_width
            scale_top = int(height * self.speed_scale_height_ratio[0])
            scale_bottom = int(height * self.speed_scale_height_ratio[1])

            # 提取区域
            speed_scale_roi = img_rgb[scale_top:scale_bottom, scale_left_col:width]

            # 增强图像
            gray = cv2.cvtColor(speed_scale_roi, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            kernel = np.ones((2, 2), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.dilate(binary, kernel, iterations=1)
            # OCR识别
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=-.0123456789'
            data = pytesseract.image_to_data(binary, config=custom_config, output_type=pytesseract.Output.DICT)

            # 解析结果
            speed_data = []
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                if text and any(c.isdigit() for c in text):
                    try:
                        # 处理负值
                        if '-' in text:
                            val = -abs(float(text.replace('-', '')))
                        else:
                            val = float(text)

                        # 获取中心位置(相对区域顶部)
                        y_pos = data['top'][i] + data['height'][i] // 2
                        speed_data.append((val, y_pos))
                    except ValueError:
                        continue

            # 按位置排序
            speed_data.sort(key=lambda x: x[1])

            if len(speed_data) >= 2:
                # 提取速度和位置
                self.speed_values = [x[0] for x in speed_data]
                self.speed_positions = [x[1] for x in speed_data]

                # 强制最后一个为负值
                self.speed_values[-1] = -abs(self.speed_values[-1])

                # 设置参考点
                self.max_speed = max(self.speed_values)
                self.min_speed = min(self.speed_values)
                self.max_speed_pos = self.speed_positions[self.speed_values.index(self.max_speed)]
                self.min_speed_pos = self.speed_positions[self.speed_values.index(self.min_speed)]

                print(f"识别到的速度刻度: {list(zip(self.speed_values, self.speed_positions))}")

            return speed_scale_roi, scale_left_col, scale_top, scale_bottom, scale_width

        except Exception as e:
            print(f"速度刻度提取错误: {str(e)}")
            return None, 0, 0, 0, 0

    def get_speed_from_relative_position(self, peak_y_relative):
        """基于相对位置计算速度"""
        if len(self.speed_values) < 2:
            return None

        # 计算速度范围和位置范围
        speed_range = self.max_speed - self.min_speed
        pos_range = abs(self.max_speed_pos - self.min_speed_pos)

        if pos_range == 0:
            return None

        # 计算峰值相对位置比例
        if self.max_speed_pos < self.min_speed_pos:  # 最高速度在上方
            position_ratio = (peak_y_relative - self.min_speed_pos) / (self.max_speed_pos - self.min_speed_pos)
        else:  # 最高速度在下方
            position_ratio = (peak_y_relative - self.max_speed_pos) / (self.min_speed_pos - self.max_speed_pos)

        # 计算速度
        peak_speed = self.min_speed + position_ratio * speed_range

        # 确保在合理范围内
        peak_speed = max(min(peak_speed, self.max_speed), self.min_speed)

        return peak_speed

    def detect_zero_velocity_line(self, img_rgb):
        """精确检测零速度线（'cm/s'标记的中间位置）"""
        try:
            # 1. 预处理 - 增强右侧刻度区域
            height, width = img_rgb.shape[:2]
            right_region = img_rgb[:, int(width * 0.7):]  # 只处理右侧30%区域

            # 转换为HSV空间提取亮色文本
            hsv = cv2.cvtColor(right_region, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))

            # 形态学处理
            kernel = np.ones((3, 3), np.uint8)
            processed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # 2. 水平投影分析找到刻度区域
            horizontal_proj = np.sum(processed, axis=1)
            threshold = 0.5 * np.max(horizontal_proj)
            scale_lines = np.where(horizontal_proj > threshold)[0]

            if len(scale_lines) == 0:
                return height * 3 // 4

            # 3. 在刻度区域使用OCR定位"cm/s"
            scale_region = right_region[min(scale_lines):max(scale_lines) + 1, :]

            # 增强OCR识别区域
            gray = cv2.cvtColor(scale_region, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            # 4. 精确OCR定位
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=cm/s0123456789-'
            data = pytesseract.image_to_data(enhanced, config=custom_config,
                                             output_type=pytesseract.Output.DICT)

            # 5. 寻找"cm/s"或数字"0"
            zero_line_pos = None
            for i in range(len(data['text'])):
                text = data['text'][i].lower().strip()
                if text == 'cm/s':
                    # 计算在原图中的绝对位置
                    zero_line_pos = min(scale_lines) + data['top'][i] + data['height'][i] // 2
                    break
                elif text == '0':
                    zero_line_pos = min(scale_lines) + data['top'][i] + data['height'][i] // 2
                    break

            # 6. 回退策略：使用刻度线中间位置
            if zero_line_pos is None and len(scale_lines) > 0:
                zero_line_pos = min(scale_lines) + (max(scale_lines) - min(scale_lines)) // 2

            return zero_line_pos if zero_line_pos is not None else height * 3 // 4

        except Exception as e:
            print(f"零速度线检测错误: {str(e)}")
            return img_rgb.shape[0] * 3 // 4



    def calculate_pressure_gradients(self, peak_speed, mean_speed):
        """计算压力梯度（基于伯努利方程简化公式：ΔP = 4v²）"""
        try:
            if peak_speed is None or mean_speed is None:
                return None, None

            # 将cm/s转换为m/s
            peak_speed_mps = peak_speed / 100.0
            mean_speed_mps = mean_speed / 100.0

            # 计算压力梯度 (mmHg)
            max_pressure_gradient = 4 * (peak_speed_mps ** 2)
            mean_pressure_gradient = 4 * (mean_speed_mps ** 2)

            # 保留2位小数，不直接取整
            max_pressure_gradient = round(max_pressure_gradient, 2)
            mean_pressure_gradient = round(mean_pressure_gradient, 2)


            return max_pressure_gradient, mean_pressure_gradient

        except Exception as e:
            print(f"压力梯度计算错误: {str(e)}")
            return None, None

    def detect_time_scale(self, img_rgb):
        """改进的时间刻度检测方法"""
        try:
            height, width = img_rgb.shape[:2]

            # 1. 扩大检测区域（底部60像素高）
            bottom_region = img_rgb[height - 60:height - 5, :]  # 从底部往上55像素

            # 2. 增强预处理
            gray = cv2.cvtColor(bottom_region, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # 自适应二值化 - 使用更敏感的阈值
            binary = cv2.adaptiveThreshold(
                blurred, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11, 2
            )

            # 3. 改进的形态学处理
            kernel_h = np.ones((1, 5), np.uint8)  # 水平方向的结构元素
            kernel_v = np.ones((3, 1), np.uint8)  # 垂直方向的结构元素

            # 先水平后垂直处理
            processed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_h)
            processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel_v)

            # 4. 改进的垂直投影分析
            projection = np.sum(processed, axis=0)
            threshold = np.max(projection) * 0.3  # 更低的阈值

            # 寻找连续区域而非单点
            above_threshold = projection > threshold
            changes = np.diff(above_threshold.astype(int))
            starts = np.where(changes == 1)[0] + 1
            ends = np.where(changes == -1)[0] + 1

            # 处理边界情况
            if above_threshold[0]:
                starts = np.insert(starts, 0, 0)
            if above_threshold[-1]:
                ends = np.append(ends, len(above_threshold))

            # 计算刻度线中心位置
            x_positions = [(s + e) // 2 for s, e in zip(starts, ends) if e - s > 1]

            if len(x_positions) < 2:
                return None, None, None

            # 5. 改进的间距计算
            diffs = np.diff(x_positions)
            median_diff = np.median(diffs)

            # 过滤异常间距
            valid_diffs = []
            valid_pairs = []
            for i in range(len(diffs)):
                if 0.5 * median_diff < diffs[i] < 2.0 * median_diff:  # 更宽松的范围
                    valid_diffs.append(diffs[i])
                    valid_pairs.append((x_positions[i], x_positions[i + 1]))

            if not valid_diffs:
                return None, None, None

            # 6. 计算平均间距
            px_per_100ms = int(np.mean(valid_diffs))

            # 7. 物理范围验证
            if not (5 <= px_per_100ms <= 150):  # 更宽松的范围
                return None, None, None

            # 8. 返回第一对有效竖线的位置和总刻度数
            first_pair = valid_pairs[0] if valid_pairs else None
            total_ticks = len(x_positions)

            return px_per_100ms, total_ticks, first_pair

        except Exception as e:
            print(f"时间刻度检测错误: {e}")
            return None, None, None

    def calculate_velocity_time_integral(self, peak_envelope_x, peak_envelope_y, img_rgb, scale_top):
        """改进的VTI计算方法"""
        try:
            if peak_envelope_x is None or len(peak_envelope_x) < 2:
                return None, None

            # 1. 检测时间刻度信息
            pixel_per_100ms, total_ticks, first_pair = self.detect_time_scale(img_rgb)

            # 如果检测失败，使用基于频谱宽度的估算
            if pixel_per_100ms is None:
                print("使用频谱宽度估算时间刻度")
                # 假设标准多普勒频谱宽度为1秒(1000ms)
                spectrum_width = peak_envelope_x[-1] - peak_envelope_x[0]
                if spectrum_width > 0:
                    pixel_per_100ms = spectrum_width / 10  # 分成10个100ms间隔
                else:
                    pixel_per_100ms = img_rgb.shape[1] / 10  # 使用图像宽度作为后备

            time_per_pixel = 0.1 / pixel_per_100ms  # 每像素对应的时间(秒)

            # 2. 计算速度值
            velocities = []
            valid_indices = []
            for i, y in enumerate(peak_envelope_y):
                speed = self.get_speed_from_relative_position(y - scale_top)
                if speed is not None and speed > 0:
                    velocities.append(speed)
                    valid_indices.append(i)

            if len(velocities) < 2:
                return None, None

            # 3. 计算VTI (cm) - 使用梯形法积分
            vti_cm = 0
            for i in range(1, len(valid_indices)):
                idx_prev = valid_indices[i - 1]
                idx_curr = valid_indices[i]

                dt = (peak_envelope_x[idx_curr] - peak_envelope_x[idx_prev]) * time_per_pixel
                avg_velocity = (velocities[i - 1] + velocities[i]) * 0.5
                vti_cm += avg_velocity * dt

            # 4. 计算波形时长
            start_col = peak_envelope_x[valid_indices[0]]
            end_col = peak_envelope_x[valid_indices[-1]]
            total_100ms_units = (end_col - start_col) / pixel_per_100ms
            time_interval = total_100ms_units * 0.1  # 转换为秒

            return round(vti_cm, 2), round(time_interval, 3)

        except Exception as e:
            print(f"VTI计算错误: {str(e)}")
            return None, None
    # def calculate_velocity_time_integral(self, current_smoothed_x, current_smoothed_y, img_rgb, scale_top,
    #                                      mean_speed_in_peak):
    #     """
    #     基于已计算的平均速度直接计算VTI（复用平均速度，避免重复积分）
    #     VTI = 平均速度（mean_speed_in_peak） × 波形持续时间（time_interval）
    #     """
    #     try:
    #         # --------------------------
    #         # 1. 验证输入有效性
    #         # --------------------------
    #         if mean_speed_in_peak is None or mean_speed_in_peak <= 0:
    #             print("错误：平均速度无效（None或非正数）")
    #             return None, None
    #
    #         if current_smoothed_x is None or len(current_smoothed_x) < 2:
    #             print("错误：时间轴数据为空或不足")
    #             return None, None
    #
    #         # 过滤无效时间点（NaN/None）
    #         valid_mask = ~np.isnan(current_smoothed_x) & ~np.isnan(current_smoothed_y)
    #         valid_x = np.array(current_smoothed_x)[valid_mask]
    #         if len(valid_x) < 2:
    #             print("有效时间点不足（至少需要2个）")
    #             return None, None
    #
    #         # --------------------------
    #         # 2. 计算波形持续时间（time_interval，单位：秒）
    #         # --------------------------
    #         # 检测时间刻度（像素→实际时间转换）
    #         pixel_per_100ms, _, _ = self.detect_time_scale(img_rgb)
    #         if pixel_per_100ms is None:
    #             print("警告：时间刻度检测失败，使用频谱宽度估算")
    #             # 估算每100ms像素数（假设频谱宽度≈1秒=10个100ms间隔）
    #             spectrum_width = valid_x[-1] - valid_x[0]
    #             pixel_per_100ms = spectrum_width / 10 if spectrum_width > 0 else img_rgb.shape[1] / 10
    #
    #         # 计算时间间隔（秒）：(终点X - 起点X) / 每100ms像素数 × 0.1秒
    #         start_x = valid_x[0]
    #         end_x = valid_x[-1]
    #         total_100ms_units = (end_x - start_x) / pixel_per_100ms
    #         time_interval = total_100ms_units * 0.1  # 转换为秒
    #
    #         if time_interval <= 0:
    #             print("错误：波形持续时间为零或负数")
    #             return None, None
    #
    #         # --------------------------
    #         # 3. 计算VTI（复用平均速度）
    #         # --------------------------
    #         vti = mean_speed_in_peak * time_interval  # VTI = 平均速度（cm/s） × 时间（s） = cm
    #
    #         return round(vti, 2), round(time_interval, 3)
    #
    #     except Exception as e:
    #         print(f"VTI计算错误: {str(e)}")
    #         return None, None
    def extract_time_info(self, img_rgb):
        """从图像最上方中间位置提取时间信息，并返回时间区域坐标"""
        try:
            height, width = img_rgb.shape[:2]

            # 定义时间信息所在的区域（顶部中间区域）
            time_region_width = width // 2
            time_region_height = 40
            x1 = (width - time_region_width) // 2
            y1 = 10
            x2 = x1 + time_region_width
            y2 = y1 + time_region_height
            time_region = img_rgb[y1:y2, x1:x2]

            # 预处理和OCR识别
            gray = cv2.cvtColor(time_region, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(binary, config=custom_config).strip()

            # 提取时间
            time_pattern = r'(\d{2}:\d{2}:\d{2})'
            match = re.search(time_pattern, text)
            if match:
                return match.group(1), (x1, y1, x2, y2)
            else:
                time_pattern = r'(\d{2}:\d{2})'
                match = re.search(time_pattern, text)
                if match:
                    return match.group(1) + ":00", (x1, y1, x2, y2)
                return "未识别到时间信息", (x1, y1, x2, y2)
        except Exception as e:
            print(f"时间信息提取错误: {str(e)}")
            return "时间提取失败", (0, 0, 0, 0)
    def trace_peak_envelope(self, upper_envelope, spectrum_roi, peak_idx, img_rgb, x_offset, y_offset, peaks,
                            second_downslopes):
        """仅追踪上包络线（强制以left_bound/right_bound为交点并添加为右端点）"""
        try:
            h_roi, w_roi = spectrum_roi.shape[:2]
            if w_roi <= 0 or h_roi <= 0:
                return None, None, []  # 返回：包络线x, 包络线y, 交点端点列表

            # ---------------------- 计算左右边界（含倒推逻辑） ----------------------
            roi_global_x_min = x_offset
            current_peak_roi = peak_idx  # 当前峰值的ROI索引

            # 1. 右边界：当前峰值右侧最近的红色竖线（peaks）→ 红色竖线交点（原有逻辑保留）
            roi_red_peaks = [global_x - roi_global_x_min for global_x in peaks
                             if 0 <= (global_x - roi_global_x_min) < w_roi]
            right_candidates = [x for x in roi_red_peaks if x > current_peak_roi]
            right_bound = min(right_candidates, default=w_roi - 1) if right_candidates else (w_roi - 1)  # 红色竖线交点的ROI索引

            # 2. 左边界：当前峰值左侧最近的蓝色竖线（second_downslopes），若无则倒推计算
            roi_blue_lines = [global_x - roi_global_x_min for global_x in second_downslopes
                              if 0 <= (global_x - roi_global_x_min) < w_roi]
            left_candidates = [x for x in roi_blue_lines if x < current_peak_roi]

            if left_candidates:
                # 正常情况：取左侧最近的蓝色竖线
                left_bound = max(left_candidates)
            else:
                # 异常情况：左侧无蓝色竖线，需倒推虚拟蓝色竖线
                # 步骤1：计算所有相邻蓝色竖线与红色竖线的间距（仅右侧存在的情况）
                valid_spacings = []
                # 遍历所有蓝色竖线，寻找其右侧最近的红色竖线
                for blue_x in roi_blue_lines:
                    red_right = [r for r in roi_red_peaks if r > blue_x]
                    if red_right:
                        min_red = min(red_right)
                        spacing = min_red - blue_x  # 蓝色到红色的间距
                        if spacing > 0:
                            valid_spacings.append(spacing)

                if valid_spacings:
                    # 步骤2：取平均间距，从当前右边界倒推虚拟左边界
                    avg_spacing = np.mean(valid_spacings)
                    virtual_left_bound = int(right_bound - avg_spacing)
                    # 确保虚拟左边界在ROI范围内且在峰值左侧
                    left_bound = max(0, min(virtual_left_bound, current_peak_roi - 1))
                else:
                    # 极端情况：无任何蓝色/红色间距，使用默认左边界（峰值左移10%）
                    left_bound = max(0, int(current_peak_roi * 0.9))

            # ---------------------- 强制提取交点坐标（重点！！） ----------------------
            # 蓝色竖线交点（left_bound）的全局坐标
            blue_intersect = (
                x_offset + left_bound,  # 全局X
                y_offset + upper_envelope[left_bound]  # 全局Y（上包络线在left_bound处的Y值）
            )
            # 红色竖线交点（right_bound）的全局坐标
            red_intersect = (
                x_offset + right_bound,  # 全局X
                y_offset + upper_envelope[right_bound]  # 全局Y（上包络线在right_bound处的Y值）
            )

            # ---------------------- 生成包络线坐标（强制包含交点） ----------------------
            # 原有逻辑：有效列筛选（根据速度阈值过滤）
            velocity_threshold_y_roi = np.interp(10.0, self.speed_values, self.speed_positions)
            valid_columns = []
            for x in range(left_bound, right_bound + 1):
                if x >= len(upper_envelope) or np.isnan(upper_envelope[x]):
                    continue
                if upper_envelope[x] < velocity_threshold_y_roi:
                    valid_columns.append(x)

            # 强制添加交点到有效列（确保包络线包含交点）
            if left_bound not in valid_columns:
                valid_columns.append(left_bound)
            if right_bound not in valid_columns:
                valid_columns.append(right_bound)
            valid_columns = sorted(valid_columns)  # 保持X递增

            if len(valid_columns) < 2:
                print(f"有效列不足（{len(valid_columns)}列）")
                return None, None, []

            # 包络线坐标（已包含两个交点）
            x_coords = [x_offset + x for x in valid_columns]
            y_coords = [y_offset + upper_envelope[x] for x in valid_columns]

            # ---------------------- 强制生成交点端点（'right'类型） ----------------------
            forced_right_endpoints = [
                (blue_intersect[0], blue_intersect[1], 'right'),  # 蓝色竖线交点→右端点
                (red_intersect[0], red_intersect[1], 'right')  # 红色竖线交点→右端点
            ]

            return np.array(x_coords), np.array(y_coords), forced_right_endpoints  # 返回：包络线+交点端点

        except Exception as e:
            print(f"包络线追踪错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None, []
    def detect_concave_endpoints(self, x_coords, y_coords, min_segment_length=3):
        """
        检测V/U形凹区间的两端端点（起点和终点）
        """
        if x_coords is None or y_coords is None:
            return []
        if isinstance(x_coords, (list, np.ndarray)) and len(x_coords) == 0:
            return []
        if isinstance(y_coords, (list, np.ndarray)) and len(y_coords) == 0:
            return []
        if len(x_coords) < min_segment_length * 2:  # 至少需要2*N个点才能形成有效区间
            return []

        # 计算一阶导数（反映上升/下降趋势，差分后长度减1）
        dx = np.diff(x_coords)
        dy = np.diff(y_coords)
        dy_dx = dy / dx  # 一阶导数（斜率）：正=上升，负=下降，0=平坦

        endpoints = []
        n = len(dy_dx)
        in_concave = False  # 是否处于凹区间内
        concave_start = 0  # 凹区间起始索引

        # 遍历一阶导数，检测"上升→下降→上升"的完整凹区间
        for i in range(1, n):
            # 1. 检测凹区间起点（左端点）：从上升/平坦转为下降（斜率由正/0变负）
            if not in_concave:
                # 前一段为上升/平坦（斜率≥-0.1），当前段转为下降（斜率＜-0.1）
                if (dy_dx[i - 1] >= -0.95) and (dy_dx[i] < -0.95):
                    in_concave = True
                    concave_start = i  # 记录凹区间起点索引（对应x_coords[i]）

            # 2. 检测凹区间终点（右端点）：从下降转为上升/平坦（斜率由负变正/0）
            else:
                # 前一段为下降（斜率＜-0.1），当前段转为上升/平坦（斜率≥-0.1）
                if (dy_dx[i - 1] < -0.95) and (dy_dx[i] >= -0.95):
                    in_concave = False
                    concave_end = i  # 记录凹区间终点索引（对应x_coords[i]）

                    # 过滤短区间（避免噪声导致的微小波动）
                    if (concave_end - concave_start) >= min_segment_length:
                        # 左端点：凹区间起点（上升→下降的转折点）
                        left_x = x_coords[concave_start]
                        left_y = y_coords[concave_start]
                        # 右端点：凹区间终点（下降→上升的转折点）
                        right_x = x_coords[concave_end]
                        right_y = y_coords[concave_end]

                        endpoints.append((left_x, left_y, 'left'))
                        endpoints.append((right_x, right_y, 'right'))

        return endpoints

    # def smooth_connect_anchors(self, peak_envelope_x, peak_envelope_y, concave_endpoints, degree=3):
    #     """
    #     使用多项式拟合连接端点
    #     """
    #     # 初始检查与数据准备（同原方法）
    #     if (peak_envelope_x is None or peak_envelope_y is None or
    #             isinstance(peak_envelope_x, (list, np.ndarray)) and len(peak_envelope_x) == 0 or
    #             isinstance(peak_envelope_y, (list, np.ndarray)) and len(peak_envelope_y) == 0):
    #         return peak_envelope_x, peak_envelope_y
    #
    #     x_arr = np.asarray(peak_envelope_x)
    #     y_arr = np.asarray(peak_envelope_y)
    #     if len(x_arr) != len(y_arr):
    #         return x_arr.tolist(), y_arr.tolist()
    #
    #     green_endpoints = sorted([p for p in concave_endpoints if p[2] == 'right'],
    #                              key=lambda p: p[0])
    #
    #     if len(green_endpoints) < 2:
    #         return x_arr.tolist(), y_arr.tolist()
    #
    #     new_x, new_y = [], []
    #
    #     # 添加第一个端点前的数据
    #     first_green_x = green_endpoints[0][0]
    #     left_mask = x_arr < first_green_x
    #     new_x.extend(x_arr[left_mask].tolist())
    #     new_y.extend(y_arr[left_mask].tolist())
    #
    #     # 处理每对相邻端点
    #     for i in range(len(green_endpoints) - 1):
    #         left_point = green_endpoints[i]
    #         right_point = green_endpoints[i + 1]
    #
    #         # 获取区间内的原始点
    #         segment_mask = (x_arr >= left_point[0]) & (x_arr <= right_point[0])
    #         segment_x = x_arr[segment_mask]
    #         segment_y = y_arr[segment_mask]
    #
    #         if len(segment_x) < 2:
    #             continue
    #
    #         # 多项式拟合
    #         coeffs = np.polyfit([left_point[0], right_point[0]],
    #                             [left_point[1], right_point[1]], degree)
    #         poly_func = np.poly1d(coeffs)
    #
    #         # 生成拟合曲线点
    #         fit_x = np.linspace(left_point[0], right_point[0], 100)
    #         fit_y = poly_func(fit_x)
    #
    #         new_x.extend(fit_x.tolist())
    #         new_y.extend(fit_y.tolist())
    #
    #     # 添加最后一个端点后的数据
    #     last_green_x = green_endpoints[-1][0]
    #     right_mask = x_arr > last_green_x
    #     new_x.extend(x_arr[right_mask].tolist())
    #     new_y.extend(y_arr[right_mask].tolist())
    #
    #     # 排序去重
    #     new_x = np.array(new_x)
    #     new_y = np.array(new_y)
    #     sorted_idx = np.argsort(new_x)
    #     new_x = new_x[sorted_idx]
    #     new_y = new_y[sorted_idx]
    #     unique_mask = np.concatenate([[True], np.diff(new_x) > 1e-6])
    #     new_x = new_x[unique_mask]
    #     new_y = new_y[unique_mask]
    #
    #     # 最终平滑滤波
    #     if len(new_y) > 5:
    #         from scipy.signal import savgol_filter
    #         new_y = savgol_filter(new_y, window_length=min(5, len(new_y)), polyorder=2)
    #
    #     return new_x.tolist(), new_y.tolist()
    #
    # def sort_and_clean(self, x_list, y_list):
    #     """
    #     辅助函数：对坐标进行排序和去重
    #     """
    #     x_arr = np.array(x_list)
    #     y_arr = np.array(y_list)
    #
    #     # 确保长度一致
    #     min_len = min(len(x_arr), len(y_arr))
    #     x_arr = x_arr[:min_len]
    #     y_arr = y_arr[:min_len]
    #
    #     # 排序
    #     sorted_idx = np.argsort(x_arr)
    #     x_arr = x_arr[sorted_idx]
    #     y_arr = y_arr[sorted_idx]
    #
    #     # 去重（保留第一个出现的值）
    #     unique_mask = np.concatenate([[True], np.diff(x_arr) > 1e-6])
    #     x_arr = x_arr[unique_mask]
    #     y_arr = y_arr[unique_mask]
    #
    #     return x_arr.tolist(), y_arr.tolist()
    # def smooth_connect_anchors(self, peak_envelope_x, peak_envelope_y, concave_endpoints):
    #     """
    #     最小二乘优化方法：在保持端点连续性的前提下进行优化
    #     """
    #     if (peak_envelope_x is None or peak_envelope_y is None or
    #             len(peak_envelope_x) == 0 or len(peak_envelope_y) == 0):
    #         return peak_envelope_x, peak_envelope_y
    #
    #     x_arr = np.asarray(peak_envelope_x)
    #     y_arr = np.asarray(peak_envelope_y)
    #
    #     # 提取关键点（端点和邻近点）
    #     key_points = []
    #     endpoints = sorted(concave_endpoints, key=lambda p: p[0])
    #
    #     for i, point in enumerate(endpoints):
    #         if i > 0 and i < len(endpoints) - 1:
    #             # 添加上下文点
    #             prev_idx = np.argmin(np.abs(x_arr - endpoints[i - 1][0]))
    #             next_idx = np.argmin(np.abs(x_arr - endpoints[i + 1][0]))
    #
    #             key_points.append((x_arr[prev_idx], y_arr[prev_idx]))
    #             key_points.append((point[0], point[1]))
    #             key_points.append((x_arr[next_idx], y_arr[next_idx]))
    #
    #     if len(key_points) < 4:
    #         return x_arr.tolist(), y_arr.tolist()
    #
    #     key_x = np.array([p[0] for p in key_points])
    #     key_y = np.array([p[1] for p in key_points])
    #
    #     # 使用二次多项式拟合
    #     A = np.vstack([key_x ** 2, key_x, np.ones(len(key_x))]).T
    #     coeffs = np.linalg.lstsq(A, key_y, rcond=None)[0]
    #
    #     # 生成拟合曲线
    #     fit_x = np.linspace(min(key_x), max(key_x), 100)
    #     fit_y = coeffs[0] * fit_x ** 2 + coeffs[1] * fit_x + coeffs[2]
    #
    #     # 合并数据
    #     mask_before = x_arr < min(key_x)
    #     mask_after = x_arr > max(key_x)
    #
    #     final_x = np.concatenate([x_arr[mask_before], fit_x, x_arr[mask_after]])
    #     final_y = np.concatenate([y_arr[mask_before], fit_y, y_arr[mask_after]])
    #
    #     # 排序去重
    #     sorted_idx = np.argsort(final_x)
    #     final_x = final_x[sorted_idx]
    #     final_y = final_y[sorted_idx]
    #
    #     return final_x.tolist(), final_y.tolist()
    # def smooth_connect_anchors(self, peak_envelope_x, peak_envelope_y, concave_endpoints):
    #     """
    #     三次样条插值方法：使用样条插值连接端点
    #     """
    #     from scipy.interpolate import CubicSpline
    #
    #     if (peak_envelope_x is None or peak_envelope_y is None or
    #             len(peak_envelope_x) == 0 or len(peak_envelope_y) == 0):
    #         return peak_envelope_x, peak_envelope_y
    #
    #     x_arr = np.asarray(peak_envelope_x)
    #     y_arr = np.asarray(peak_envelope_y)
    #
    #     # 提取并排序端点
    #     endpoints = sorted([p for p in concave_endpoints if p[2] in ['left', 'right']],
    #                        key=lambda p: p[0])
    #     endpoint_x = [p[0] for p in endpoints]
    #     endpoint_y = [p[1] for p in endpoints]
    #
    #     if len(endpoint_x) < 4:  # 至少需要4个点进行样条插值
    #         return x_arr.tolist(), y_arr.tolist()
    #
    #     # 创建样条插值函数
    #     try:
    #         cs = CubicSpline(endpoint_x, endpoint_y, bc_type='natural')
    #     except:
    #         return x_arr.tolist(), y_arr.tolist()
    #
    #     # 生成插值点
    #     new_x = np.linspace(min(endpoint_x), max(endpoint_x),
    #                         len(endpoint_x) * 10)
    #     new_y = cs(new_x)
    #
    #     # 合并原始数据和插值数据
    #     mask_before = x_arr < min(endpoint_x)
    #     mask_after = x_arr > max(endpoint_x)
    #
    #     final_x = np.concatenate([x_arr[mask_before], new_x, x_arr[mask_after]])
    #     final_y = np.concatenate([y_arr[mask_before], new_y, y_arr[mask_after]])
    #
    #     # 排序去重
    #     sorted_idx = np.argsort(final_x)
    #     final_x = final_x[sorted_idx]
    #     final_y = final_y[sorted_idx]
    #
    #     return final_x.tolist(), final_y.tolist()
    def smooth_connect_anchors(self, peak_envelope_x, peak_envelope_y, concave_endpoints):
        """
        修改版：贝塞尔曲线  在每对相邻的绿色圆形端点之间生成平滑曲线
        """
        # 输入有效性检查（修复数组判断问题）
        if (peak_envelope_x is None or peak_envelope_y is None or
                isinstance(peak_envelope_x, (list, np.ndarray)) and len(peak_envelope_x) == 0 or
                isinstance(peak_envelope_y, (list, np.ndarray)) and len(peak_envelope_y) == 0):
            return peak_envelope_x, peak_envelope_y

        # 转换为数组并检查长度
        x_arr = np.asarray(peak_envelope_x)
        y_arr = np.asarray(peak_envelope_y)
        if len(x_arr) != len(y_arr):
            print(f"警告：X和Y长度不匹配（X:{len(x_arr)}, Y:{len(y_arr)}）")
            return x_arr.tolist(), y_arr.tolist()

        # 提取并排序绿色圆形端点（按X坐标）
        green_endpoints = sorted([p for p in concave_endpoints if p[2] == 'right'],
                                 key=lambda p: p[0])

        if len(green_endpoints) < 2:
            return x_arr.tolist(), y_arr.tolist()

        # 生成三次贝塞尔曲线函数
        def cubic_bezier(p0, p1, p2, p3, num_points=50):
            """生成三次贝塞尔曲线"""
            t = np.linspace(0, 1, num_points)
            x = (1 - t) ** 3 * p0[0] + 3 * (1 - t) ** 2 * t * p1[0] + 3 * (1 - t) * t ** 2 * p2[0] + t ** 3 * p3[0]
            y = (1 - t) ** 3 * p0[1] + 3 * (1 - t) ** 2 * t * p1[1] + 3 * (1 - t) * t ** 2 * p2[1] + t ** 3 * p3[1]
            return x, y

        # 初始化结果数组
        new_x, new_y = [], []

        # 添加第一个绿色端点左侧的原始包络线
        first_green_x = green_endpoints[0][0]
        left_mask = x_arr < first_green_x
        new_x.extend(x_arr[left_mask].tolist())
        new_y.extend(y_arr[left_mask].tolist())

        # 处理每对相邻的绿色端点
        for i in range(len(green_endpoints) - 1):
            left_point = green_endpoints[i]
            right_point = green_endpoints[i + 1]

            # 获取两个端点之间的原始包络线点
            segment_mask = (x_arr >= left_point[0]) & (x_arr <= right_point[0])
            segment_x = x_arr[segment_mask]
            segment_y = y_arr[segment_mask]

            if len(segment_x) < 2:
                continue

            # 计算控制点
            # 控制点1：左端点右侧的原始包络线点
            ctrl1_x = left_point[0] + (right_point[0] - left_point[0]) * 0.3
            ctrl1_y = left_point[1] - (left_point[1] - np.min(segment_y)) * 0.5

            # 控制点2：右端点左侧的原始包络线点
            ctrl2_x = right_point[0] - (right_point[0] - left_point[0]) * 0.3
            ctrl2_y = right_point[1] - (right_point[1] - np.min(segment_y)) * 0.5

            # 生成贝塞尔曲线
            bezier_x, bezier_y = cubic_bezier(
                (left_point[0], left_point[1]),
                (ctrl1_x, ctrl1_y),
                (ctrl2_x, ctrl2_y),
                (right_point[0], right_point[1]),
                num_points=100
            )

            # 添加平滑曲线
            new_x.extend(bezier_x.tolist())
            new_y.extend(bezier_y.tolist())

        # 添加最后一个绿色端点右侧的原始包络线
        last_green_x = green_endpoints[-1][0]
        right_mask = x_arr > last_green_x
        new_x.extend(x_arr[right_mask].tolist())
        new_y.extend(y_arr[right_mask].tolist())

        # 转换为数组并排序去重
        new_x = np.array(new_x)
        new_y = np.array(new_y)
        sorted_idx = np.argsort(new_x)
        new_x = new_x[sorted_idx]
        new_y = new_y[sorted_idx]
        unique_mask = np.concatenate([[True], np.diff(new_x) > 1e-6])
        new_x = new_x[unique_mask]
        new_y = new_y[unique_mask]

        # 最终平滑滤波
        if len(new_y) > 5:
            from scipy.signal import savgol_filter
            new_y = savgol_filter(new_y, window_length=5, polyorder=2)
        return new_x.tolist(), new_y.tolist()

    def calculate_average_velocity(self, double_smoothed_x, double_smoothed_y, img_rgb, scale_top):
        """
        计算包络线与零刻度线包围的补全后波形的平均速度（基于面积积分）
        """
        try:
            if double_smoothed_x is None or double_smoothed_y is None:
                print("错误：包络线数据为空")
                return None

            # 过滤NaN值，转换为numpy数组便于处理
            valid_mask = ~np.isnan(double_smoothed_x) & ~np.isnan(double_smoothed_y)
            valid_x = np.array(double_smoothed_x)[valid_mask]  # 时间轴（X轴，单位：ms或像素）
            valid_y = np.array(double_smoothed_y)[valid_mask]  # 包络线Y坐标（ROI内像素）

            if len(valid_x) < 5:  # 至少需要5个点构成连续波形
                print(f"有效包络线点不足：{len(valid_x)}个点")
                return None

            zero_line_global_y = self.detect_zero_velocity_line(img_rgb)
            if zero_line_global_y is None:
                print("警告：零刻度线检测失败，使用速度刻度中间值替代")
                zero_line_global_y = np.interp(0, self.speed_values, self.speed_positions) + scale_top
            velocities = np.array([
                self.get_speed_from_relative_position(y - scale_top)
                for y in valid_y
            ])
            time_points = valid_x
            upper_velocities = velocities  # 上沿：包络线速度
            lower_velocities = np.zeros_like(upper_velocities)  # 下沿：零刻度线（速度=0）

            # 合并上沿和下沿（反转下沿顺序，形成闭合多边形）
            all_time = np.concatenate([time_points, time_points[::-1]])  # 时间轴：正序+倒序
            all_velocities = np.concatenate([upper_velocities, lower_velocities[::-1]])  # 速度轴：上沿+下沿倒序
            vti = np.trapz(all_velocities, all_time) * 0.001  # VTI（速度时间积分），单位：cm
            waveform_duration = (time_points[-1] - time_points[0]) * 0.001  # 单位：s

            if waveform_duration <= 0:
                print("错误：波形持续时间为零")
                return None

            average_velocity = vti / waveform_duration  # 平均速度 = 面积 / 时间，单位：cm/s

            # 保留2位小数，确保非负（速度应为正值）
            return round(max(average_velocity, 0), 2)

        except Exception as e:
            print(f"平均速度计算错误: {str(e)}")
            return None
    # def calculate_average_velocity(self, double_smoothed_x, double_smoothed_y, img_rgb, scale_top):
    #     """
    #     计算平滑峰值波形的包络线与零刻度线之间包围的波形的平均速度
    #     """
    #     try:
    #         # 1. 检测零刻度线位置（全局Y坐标）
    #         zero_line_pos = self.detect_zero_velocity_line(img_rgb)
    #         if zero_line_pos is None:
    #             print("无法检测零刻度线位置")
    #             return None
    #
    #         # 2. 提取有效包络线点（去除NaN值）
    #         valid_indices = [i for i, y in enumerate(double_smoothed_y) if not np.isnan(y)]
    #         if len(valid_indices) < 2:
    #             print("有效包络线点不足")
    #             return None
    #
    #         valid_x = [double_smoothed_x[i] for i in valid_indices]
    #         valid_y = [double_smoothed_y[i] for i in valid_indices]
    #
    #         # 3. 计算每个时间点的速度值（包络线与零刻度线之间）
    #         velocities = []
    #         for x, y in zip(valid_x, valid_y):
    #             # 计算包络线点到零刻度线的距离（像素）
    #             distance_to_zero = zero_line_pos - y
    #
    #             # 计算速度值（基于速度刻度插值）
    #             speed = self.get_speed_from_relative_position(y - scale_top)
    #             if speed is not None and speed > 0:  # 只保留正速度
    #                 velocities.append(speed)
    #
    #         if not velocities:
    #             print("未计算出有效速度值")
    #             return None
    #
    #         # 4. 计算平均速度
    #         average_velocity = np.mean(velocities)
    #         return round(average_velocity, 2)
    #
    #     except Exception as e:
    #         print(f"平均速度计算错误: {str(e)}")
    #         return None

    # def calculate_average_velocity(self, x_coords, y_coords, img_rgb, scale_top):
    #     """
    #     优化版平均速度计算
    #     使用加权平均
    #     """
    #     try:
    #         # 1. 获取时间刻度信息
    #         px_per_100ms, _, _ = self.detect_time_scale(img_rgb)
    #         if px_per_100ms is None:
    #             px_per_100ms = (x_coords[-1] - x_coords[0]) / 10  # 默认假设1秒宽度
    #
    #         # 2. 转换速度值
    #         velocities = []
    #         valid_indices = []
    #
    #         for i, y in enumerate(y_coords):
    #             speed = self.get_speed_from_relative_position(y - scale_top)
    #             if speed is not None and speed > 0:
    #                 velocities.append(speed)
    #                 valid_indices.append(i)
    #
    #         if len(velocities) < 3:
    #             return None
    #
    #         # 3. 计算加权VTI (速度高的部分权重更大)
    #         total_weighted_area = 0
    #         total_time = 0
    #         weights = np.array(velocities) / max(velocities)  # 速度越高权重越大
    #
    #         for i in range(1, len(valid_indices)):
    #             idx_prev = valid_indices[i - 1]
    #             idx_curr = valid_indices[i]
    #
    #             dt = (x_coords[idx_curr] - x_coords[idx_prev]) / px_per_100ms * 0.1  # 秒
    #             avg_velocity = (velocities[i - 1] + velocities[i]) / 2
    #             weight = (weights[i - 1] + weights[i]) / 2
    #
    #             total_weighted_area += avg_velocity * dt * weight
    #             total_time += dt * weight
    #
    #         if total_time <= 0:
    #             return None
    #
    #         # 4. 计算加权平均速度
    #         weighted_avg_velocity = total_weighted_area / total_time
    #         return round(max(weighted_avg_velocity, 0), 2)
    #
    #     except Exception as e:
    #         print(f"平均速度计算错误: {str(e)}")
    #         return None
    def detect_spectral_line(self, img_rgb):
        """改进的绿色频谱线区域检测方法"""
        try:
            # 转换到HSV颜色空间
            hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

            # 定义绿色范围（调整更精确的范围）
            lower_green = np.array([40, 40, 80])  # 提高饱和度下限
            upper_green = np.array([100, 255, 255])
            mask = cv2.inRange(hsv, lower_green, upper_green)

            # 形态学处理（更适合细长频谱线）
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.dilate(mask, None, iterations=2)

            # 查找轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                print("未检测到频谱线区域")
                return None

            # 改进的轮廓选择策略（考虑长宽比和位置）
            valid_contours = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)

                # 过滤条件：
                # 1. 长宽比 > 3（频谱线通常是细长的）
                # 2. 宽度 > 20像素（避免小噪声）
                # 3. 高度 > 5像素
                aspect_ratio = w / max(h, 1e-5)
                if aspect_ratio > 3 and w > 20 and h > 5:
                    valid_contours.append((cnt, aspect_ratio * w))  # 用长宽比*宽度作为评分

            if not valid_contours:
                print("未找到符合要求的频谱线轮廓")
                return None

            # 选择评分最高的轮廓
            main_contour = max(valid_contours, key=lambda x: x[1])[0]
            x, y, w, h = cv2.boundingRect(main_contour)
            print(f"频谱线区域: x={x}, y={y}, w={w}, h={h}")
            return x, y - 1, w, h + 1

        except Exception as e:
            print(f"频谱线检测错误: {str(e)}")
            return None
    # def analyze_spectral_peaks(self, img_rgb, spectral_region):
    #     """自适应动态阈值处理"""
    #     try:
    #         # 参数校验和初始化
    #         if spectral_region is None:
    #             print("频谱区域为空")
    #             return np.array([]), np.array([]), []
    #
    #         x, y, w, h = map(int, spectral_region[:4])
    #         img_h, img_w = img_rgb.shape[:2]
    #
    #         # ROI截取（带边界检查和最小尺寸保证）
    #         y1, y2 = max(y, 0), min(y + max(h, 10), img_h)  # 最小高度10像素
    #         x1, x2 = max(x, 0), min(x + max(w, 50), img_w)  # 最小宽度50像素
    #         spectral_roi = img_rgb[y1:y2, x1:x2]
    #
    #         actual_w = spectral_roi.shape[1]
    #         actual_h = spectral_roi.shape[0]
    #         if actual_w < 30 or actual_h < 5:
    #             print(f"ROI尺寸过小: {actual_w}x{actual_h}")
    #             return np.array([]), np.array([]), []
    #
    #         # 信号提取和维度处理（确保1D数组）
    #         try:
    #             green_channel = spectral_roi[:, :, 1]
    #             # 多行采样取平均，然后展平为1D数组
    #             center_rows = [
    #                 green_channel[i, :].flatten()
    #                 for i in [actual_h // 3, actual_h // 2, 2 * actual_h // 3]
    #             ]
    #             avg_row = np.mean(center_rows, axis=0).astype(np.float32)
    #             if avg_row.ndim > 1:
    #                 avg_row = avg_row.squeeze()  # 确保变为1D
    #         except Exception as e:
    #             print(f"信号提取错误: {str(e)}")
    #             return np.array([]), np.array([]), []
    #
    #
    #         kernel_size = max(3, int(actual_w * 0.03) | 1)  # 奇数内核
    #         smoothed = cv2.GaussianBlur(avg_row, (kernel_size, 1), 0).flatten()  # 再次确保1D
    #
    #         # 标准化信号（防止除零）
    #         norm_signal = (smoothed - np.min(smoothed)) / (np.max(smoothed) - np.min(smoothed) + 1e-8)
    #         norm_signal = norm_signal.squeeze()  # 最终确保1D
    #
    #         # 峰值检测（带维度检查）
    #         if norm_signal.ndim != 1:
    #             print(f"信号维度错误: {norm_signal.shape}")
    #             return np.array([]), np.array([]), []
    #
    #         peaks = self._robust_peak_detection(norm_signal, actual_w)
    #
    #         # 结果后处理
    #         if len(peaks) < 2:
    #             print(f"峰值不足: {len(peaks)}个")
    #             return np.array([]), np.array([]), []
    #
    #         global_peaks, blue_lines = self._process_results(peaks, norm_signal, actual_w, x1, x)
    #
    #         return global_peaks, smoothed, blue_lines
    #
    #     except Exception as e:
    #         print(f"分析出错: {str(e)}")
    #         return np.array([]), np.array([]), []
    # def _process_results(self, peaks, signal, window_width, x1, x_offset):
    #     """处理检测结果"""
    #     # 按高度过滤
    #     peak_heights = signal[peaks]
    #     valid_peaks = peaks[peak_heights > np.percentile(peak_heights, 30)]
    #
    #     # 全局坐标转换
    #     global_peaks = (x1 + valid_peaks - x_offset).astype(int)
    #
    #     # 计算标记线
    #     blue_lines = []
    #     sorted_peaks = sorted(valid_peaks)
    #     min_dist = max(10, window_width // 10)
    #
    #     for i in range(len(sorted_peaks) - 1):
    #         dist = sorted_peaks[i + 1] - sorted_peaks[i]
    #         if dist > min_dist:
    #             pos = x1 + sorted_peaks[i] + int(dist / 3) - x_offset
    #             blue_lines.append(pos)
    #
    #     return global_peaks, blue_lines
    #
    # def _robust_peak_detection(self, signal, window_width):
    #     """鲁棒的峰值检测（确保1D输入）"""
    #     assert signal.ndim == 1, "信号必须是1D数组"
    #
    #     # 动态参数设置
    #     min_dist = max(5, window_width // 15)
    #     median_val = np.median(signal)
    #     dynamic_thresh = median_val + 0.3 * (np.max(signal) - median_val)
    #
    #     try:
    #         from scipy.signal import find_peaks
    #         peaks, _ = find_peaks(
    #             signal,
    #             height=max(0.2, dynamic_thresh),
    #             distance=min_dist,
    #             prominence=max(0.1, median_val * 0.5),
    #             width=2
    #         )
    #         return peaks
    #     except ImportError:
    #         # 备用算法
    #         peaks = []
    #         for i in range(1, len(signal) - 1):
    #             if signal[i] > dynamic_thresh and signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
    #                 peaks.append(i)
    #         return np.array(peaks)
    # def analyze_spectral_peaks(self, img_rgb, spectral_region):
    #     """HBVigor"""
    #     x, y, w, h = spectral_region[:4]
    #     spectral_roi = img_rgb[y:y + h, x:x + w]
    #     gray = cv2.cvtColor(spectral_roi, cv2.COLOR_RGB2GRAY)
    #     center_row = gray[h // 2, :].flatten().astype(np.float32)
    #
    #     # Step 1: 初始峰值检测（HBVigor方法）
    #     peaks = self.detect_peaks_with_hbv(center_row, min_height=30, min_slope=0.5, min_interval=20)
    #
    #     # Step 2: 使用HRV方法优化峰值间隔
    #     refined_peaks = self.refine_peaks_with_hrv(peaks, center_row, max_interval_variation=10)
    #
    #     # Step 3: 转换到全局坐标
    #     global_peaks = [x + peak for peak in refined_peaks]
    #
    #     return np.array(global_peaks), center_row, np.array(global_peaks) + 5
    #
    # def refine_peaks_with_hrv(self,peaks, signal, max_interval_variation=10):
    #     refined_peaks = []
    #     intervals = np.diff(peaks)  # 计算峰值间隔
    #
    #     # 过滤间隔异常的峰值（类似NN50/pNN50）
    #     for i in range(1, len(peaks)):
    #         if abs(intervals[i - 1] - np.median(intervals)) < max_interval_variation:
    #             refined_peaks.append(peaks[i])
    #
    #     # 计算RMSSD（评估峰值间隔稳定性）
    #     rmssd = np.sqrt(np.mean(np.square(np.diff(intervals))))
    #     print(f"Peak interval stability (RMSSD): {rmssd:.2f} pixels")
    #
    #     return np.array(refined_peaks)
    # def detect_peaks_with_hbv(self, signal, min_height=30, min_slope=0.5, min_interval=20):
    #     """HBVigor"""
    #     peaks = []
    #     for i in range(1, len(signal) - 1):
    #         # 检查是否高于阈值（幅度）
    #         if signal[i] < min_height:
    #             continue
    #
    #         # 检查是否处于上升沿或下降沿（斜率）
    #         slope_prev = signal[i] - signal[i - 1]
    #         slope_next = signal[i] - signal[i + 1]
    #         if slope_prev < min_slope or slope_next < min_slope:
    #             continue
    #
    #         # 检查是否与上一个峰值间隔足够（一致性）
    #         if peaks and (i - peaks[-1]) < min_interval:
    #             continue
    #
    #         peaks.append(i)
    #
    #     return np.array(peaks)  # 修正：应该在循环结束后返回
    def analyze_spectral_peaks(self, img_rgb, spectral_region):
        """our高斯平滑去噪动态阈值"""
        try:
            x, y, w, h = spectral_region[:4]  # 频谱区域在原图的左上角坐标(x,y)及宽高(w,h)
            img_h, img_w = img_rgb.shape[:2]  # 获取原图尺寸，用于边界检查

            # ---------------------- 1. 修复ROI截取（避免超出图像边界）----------------------
            # 截取ROI时限制边界，确保实际宽度/高度正确
            spectral_roi = img_rgb[
                           max(y, 0):min(y + h, img_h),  # 高度范围（不超出图像）
                           max(x, 0):min(x + w, img_w)  # 宽度范围（不超出图像）
                           ]
            actual_w = spectral_roi.shape[1]  # 实际ROI宽度（可能小于w，修复坐标偏移的关键）
            actual_h = spectral_roi.shape[0]
            if actual_w == 0 or actual_h == 0:
                print("频谱ROI为空（超出图像边界）")
                return np.array([]), np.array([]), []  # 返回空值

            # ---------------------- 2. 提取绿色通道和中心线 ----------------------
            green_channel = spectral_roi[:, :, 1]  # 频谱图通常用绿色通道承载信号
            center_row = green_channel[actual_h // 2, :].flatten().astype(np.float32)  # 取ROI中间行

            # ---------------------- 3. 峰值检测（保留平滑和阈值逻辑）----------------------
            smoothed = cv2.GaussianBlur(center_row, (21, 21), 0).flatten()  # 高斯平滑去噪
            median_val = np.median(smoothed)  # 基于中位数设定峰值阈值，增强鲁棒性
            peaks, properties = find_peaks(
                smoothed,
                height=(30, 100),  # 峰值高度阈值（动态适应信号强度）
                distance=actual_w // 8,  # 峰值最小间距（避免过近噪声峰，ROI宽度的1/8）
                prominence=median_val * 0.7,  # 峰值突出度（确保是显著峰值）
                width=2  # 最小峰宽（过滤尖锐噪声）
            )
            if len(peaks) < 2:
                print(f"有效峰值不足（仅检测到{len(peaks)}个，需至少2个）")
                return np.array([]), np.array([]), []

            # ---------------------- 4. 核心：转换峰值为全局坐标 ----------------------
            # peaks是ROI内的相对X坐标，需转换为原图的全局X坐标（x + peak）
            global_peaks = [x + peak for peak in peaks]  # 逐个转换峰值坐标
            global_peaks = np.array(global_peaks)  # 转为数组，方便后续处理

            # ---------------------- 5. 计算相邻峰值前1/3位置（蓝色竖线）----------------------
            blue_lines = []
            sorted_peaks = sorted(peaks)  # 按ROI内X坐标排序（确保从左到右处理）

            for i in range(len(sorted_peaks) - 1):
                peak_left_roi = sorted_peaks[i]  # 左侧峰值（ROI内相对坐标）
                peak_right_roi = sorted_peaks[i + 1]  # 右侧峰值（ROI内相对坐标）
                peak_distance = peak_right_roi - peak_left_roi  # 两峰值在ROI内的间距

                # 过滤过近峰值（间距小于ROI宽度的1/20，视为噪声）
                if peak_distance < max(20, actual_w // 20):
                    continue

                # 计算前1/3位置（ROI内相对坐标）
                third_pos_roi = int(peak_left_roi + peak_distance * (1 / 3))
                # 转换为全局坐标（原图X坐标）
                third_pos_global = x + third_pos_roi
                blue_lines.append(third_pos_global)

            # 返回：全局坐标的峰值、平滑曲线、全局坐标的蓝色竖线
            return global_peaks, smoothed, blue_lines

        except Exception as e:
            print(f"频谱峰值分析错误: {str(e)}")
            return np.array([]), np.array([]), []
    # def analyze_spectral_peaks(self, img_rgb, spectral_region):
    #     """离散小波变换"""
    #     try:
    #         x, y, w, h = spectral_region[:4]
    #         img_h, img_w = img_rgb.shape[:2]
    #
    #         # ---------------------- 1. ROI截取（保持原逻辑）----------------------
    #         spectral_roi = img_rgb[
    #                        max(y, 0):min(y + h, img_h),
    #                        max(x, 0):min(x + w, img_w)
    #                        ]
    #         actual_w, actual_h = spectral_roi.shape[1], spectral_roi.shape[0]
    #         if actual_w == 0 or actual_h == 0:
    #             print("频谱ROI为空")
    #             return np.array([]), np.array([]), []
    #
    #         # ---------------------- 2. 提取信号（增强鲁棒性）----------------------
    #         green_channel = spectral_roi[:, :, 1]
    #         # 多行平均（原单行→3行平均，减少噪声影响）
    #         signal = np.mean([
    #             green_channel[actual_h // 3, :],
    #             green_channel[actual_h // 2, :],
    #             green_channel[2 * actual_h // 3, :]
    #         ], axis=0).astype(np.float32)
    #
    #         # ---------------------- 3. 离散小波去噪（核心改进）----------------------
    #         # 步骤1：小波分解（选择适合平滑信号的小波基和层数）
    #         wavelet = 'db4'  # 常用Daubechies小波，db4适合平滑信号
    #         level = 3  # 分解层数（根据信号长度动态调整）
    #         coeffs = pywt.wavedec(signal, wavelet, level=level)  # 多层分解
    #
    #         # 步骤2：阈值处理高频系数（去除噪声）
    #         sigma = np.median(np.abs(coeffs[-1])) / 0.6745  # 估计噪声标准差（通用公式）
    #         threshold = sigma * np.sqrt(2 * np.log(len(signal)))  # 通用阈值公式
    #
    #         # 对所有高频系数应用软阈值（保留边缘特征）
    #         denoised_coeffs = coeffs.copy()
    #         for i in range(1, len(coeffs)):  # 跳过近似系数(coeffs[0])
    #             denoised_coeffs[i] = pywt.threshold(
    #                 coeffs[i],
    #                 value=threshold,
    #                 mode='soft'  # 软阈值：系数绝对值<阈值则置0，否则减去阈值
    #             )
    #
    #         # 步骤3：小波重构（从去噪后的系数恢复信号）
    #         denoised_signal = pywt.waverec(denoised_coeffs, wavelet)
    #         # 确保重构信号长度与原始一致（小波分解可能有长度差异）
    #         denoised_signal = denoised_signal[:len(signal)]
    #
    #         # ---------------------- 4. 基于小波特征的峰值检测 ----------------------
    #         # 计算小波能量作为峰值显著性指标（增强峰值特征）
    #         wavelet_energy = np.sum([np.sum(np.square(c)) for c in coeffs[1:]], axis=0)
    #         energy_norm = wavelet_energy / np.max(wavelet_energy)  # 归一化能量
    #
    #         # 动态阈值：结合小波能量和信号幅值
    #         median_val = np.median(denoised_signal)
    #         height_thresh = max(median_val * 1.2, np.percentile(denoised_signal, 80))  # 80%分位数
    #
    #         peaks, properties = find_peaks(
    #             denoised_signal,
    #             height=height_thresh,  # 基于小波去噪信号的高度阈值
    #             distance=actual_w // 10,  # 峰值最小间距（根据ROI宽度调整）
    #             prominence=energy_norm * 0.5,  # 结合小波能量的突出度阈值
    #             width=3  # 最小峰宽（过滤窄带噪声）
    #         )
    #
    #         # ---------------------- 5. 后处理与结果返回 ----------------------
    #         if len(peaks) < 2:
    #             print(f"有效峰值不足（检测到{len(peaks)}个）")
    #             return np.array([]), denoised_signal, []
    #
    #         # 转换为全局坐标
    #         global_peaks = np.array([x + peak for peak in peaks])
    #
    #         # 计算蓝色标记线（保持原逻辑）
    #         blue_lines = []
    #         sorted_peaks = sorted(peaks)
    #         for i in range(len(sorted_peaks) - 1):
    #             dist = sorted_peaks[i + 1] - sorted_peaks[i]
    #             if dist > max(20, actual_w // 20):
    #                 third_pos_global = x + sorted_peaks[i] + int(dist * (1 / 3))
    #                 blue_lines.append(third_pos_global)
    #
    #         return global_peaks, denoised_signal, blue_lines
    #
    #     except Exception as e:
    #         print(f"小波变换分析错误: {str(e)}")
    #         return np.array([]), np.array([]), []
    def visualize(self, image_path):
        """可视化分析结果（包含峰值检测和速度刻度线框选）并计算峰值速度"""
        global peak_speed, mean_speed_in_peak, zero_line_global, max_pressure_gradient, mean_pressure_gradient, peak_envelope_x, peak_envelope_y, x_sl
        img_gray, img_rgb = self.load_image(image_path, return_original=True)
        if img_gray is None or img_rgb is None:
            return False
        time_info = self.extract_time_info(img_rgb)
        print(f"图像时间: {time_info}")
        # 获取图像尺寸
        height, width = img_gray.shape[:2]
        y_start = int(height * 0.08)  # 10%高度
        y_end = int(height * 0.4)  # 40%高度
        x_start = int(width * 0.3)  # 20%宽度
        x_end = int(width * 0.7)  # 80%宽度

        # 绘制矩形框e
        cv2.rectangle(img_rgb,
                      (x_start, y_start),
                      (x_end, y_end),
                      color=(255, 0, 255),  # 品红色
                      thickness=3)
        # 定位频谱区域

        x, y, w, h_roi = self.locate_spectrum(img_gray)
        spectrum_roi = img_gray[y:y+ h_roi, x:x + w]
        cv2.rectangle(img_rgb,
                      (x, y),
                      (x + w, y + h_roi),
                      (255, 255, 0),
                      3)
        upper_envelope, lower_envelope= self.extract_envelopes(spectrum_roi,smooth_method='moving_avg')#,,,

        # 找到上包络线峰值位置
        valid_upper = upper_envelope[~np.isnan(upper_envelope)]
        if len(valid_upper) == 0:
            print("未检测到有效包络线")
            return False

        # 上包络线最小值点对应峰值位置（最靠近顶部）
        peak_y_idx = np.nanargmin(upper_envelope)
        peak_y_roi = upper_envelope[peak_y_idx]
        peak_x = x + peak_y_idx  # 峰值在原图x坐标
        peak_global_y = y + peak_y_roi  # 峰值在原图y坐标

        # 检测频谱线区域
        spectral_region = self.detect_spectral_line(img_rgb)
        if spectral_region:
            x_sl, y_sl, w_sl, h_sl = spectral_region
            spectral_roi = img_rgb[y_sl:y_sl + h_sl, x_sl:x_sl + w_sl]
            peaks, smoothed,second_downslopes= self.analyze_spectral_peaks(img_rgb, spectral_region)
            # height_increase_ratio = 2.7  # 高度增加比例（例如1.5表示增加50%）
            # new_h_sl = int(h_sl * height_increase_ratio)  # 新高度
            # # 向上偏移起始Y坐标，使框同时向上和向下扩展（保持中心大致不变）
            # new_y_sl = int(y_sl - (new_h_sl - h_sl) / 2)  # 新起始Y坐标

            # 绘制调整后的框
            cv2.rectangle(img_rgb,
                          (x_sl, y_sl),  # 使用新的Y坐标
                          (x_sl + w_sl, y_sl + h_sl),  # 使用新高度
                          color=(0, 255, 0),  # 绿色
                          thickness=3)

        else:
            peaks = []
            second_downslopes = []

        # 提取速度刻度区域并识别速度值
        speed_scale_roi, scale_left_col, scale_top, scale_bottom, scale_width = self.extract_speed_scale_region(img_rgb)
        speed_scale_coords = None  # 存储速度刻度区域坐标

        if speed_scale_roi is not None:
            # 保存速度刻度区域坐标
            speed_scale_coords = (scale_left_col, scale_top, scale_width, scale_bottom - scale_top)
            print("识别到的速度刻度值:", self.speed_values)
            print("对应的垂直位置:", self.speed_positions)

            # 计算峰值对应的速度值
            peak_y_relative = peak_global_y - scale_top  # 相对于刻度区域顶部的位置
            peak_speed = self.get_speed_from_relative_position(peak_y_relative)

            zero_line_global = y + self.detect_zero_velocity_line(img_rgb[y:y + h_roi, x:x + w])
            peaks_left_shifted = np.clip(peaks - 15, 0, img_rgb.shape[1] - 1)
            peak_envelope_x, peak_envelope_y, right_endpoints = self.trace_peak_envelope(
                upper_envelope, spectrum_roi, peak_y_idx, img_rgb, x, y, peaks_left_shifted, second_downslopes
            )
            part_endpoints = self.detect_concave_endpoints(
                peak_envelope_x, peak_envelope_y, min_segment_length=5
            )
            peak_point = (peak_x, peak_global_y, 'right')
            concave_endpoints = right_endpoints.copy()
            for ep in part_endpoints:
                # 避免重复（如果自动检测到交点，跳过）
                if not any(abs(ep[0] - fe[0]) < 2 for fe in right_endpoints):  # X坐标差<2视为同一端点
                    concave_endpoints.append(ep)

            smoothed_x, smoothed_y = self.smooth_connect_anchors(
                peak_envelope_x, peak_envelope_y, concave_endpoints
            )
            current_smoothed_x, current_smoothed_y = smoothed_x.copy(), smoothed_y.copy()

            for iter in range(1, 7):
                # 1. 检测当前曲线的凹区间端点
                concave_pts = self.detect_concave_endpoints(
                    current_smoothed_x,
                    current_smoothed_y,
                    min_segment_length=3  # 保持与原方法一致的阈值
                )
                if iter <= 2 and right_endpoints:  # 确保 right_endpoints 非空
                    first_endpoint = right_endpoints[0]  # 获取第一个端点
                    # 检查是否已存在（避免重复添加）
                    if not any(abs(first_endpoint[0] - ep[0]) < 2 for ep in concave_pts):
                        concave_pts.append(first_endpoint)
                    if not any(abs(peak_point[0] - ep[0]) < 2 for ep in concave_pts):
                        concave_pts.append(peak_point)
                current_smoothed_x, current_smoothed_y = self.smooth_connect_anchors(
                    current_smoothed_x,
                    current_smoothed_y,
                    concave_pts
                )
            mean_speed_in_peak = self.calculate_average_velocity(current_smoothed_x, current_smoothed_y, img_rgb,
                                                                 scale_top)
            max_pressure_gradient, mean_pressure_gradient = self.calculate_pressure_gradients(peak_speed,
                                                                                              mean_speed_in_peak)
            vti, time_interval = self.calculate_velocity_time_integral(
                peak_envelope_x, peak_envelope_y, img_rgb, scale_top)
        # 绘制分析结果
        plt.figure(figsize=(15, 8))
        plt.imshow(img_rgb)
        if spectral_region and len(peaks) > 0:
            for idx, peak in enumerate(peaks):
                abs_x = peak
                # plt.axvline(x=abs_x, color='red', linestyle='-', linewidth=1.5, alpha=0.8 if idx == 0 else None)# label='有效峰值'
                roi_x = abs_x - x_sl  # 将全局X坐标转换为ROI内的局部X坐标
                if 0 <= roi_x < w_sl:  # 确保X坐标在ROI范围内
                    green_channel = spectral_roi[:, roi_x, 1]  # 提取ROI中第roi_x列的绿色通道值（0~255）
                    peak_roi_y = np.argmax(green_channel)  # 获取绿色值最大的像素行索引（ROI内局部坐标）
                    peak_global_y = y_sl + peak_roi_y- 9  # 频谱区域起点Y + ROI内Y坐标
                    # plt.scatter(abs_x, peak_global_y,
                    #             color='red',  # 黄色填充
                    #             s=140,  # 点大小（可根据图像分辨率调整）
                    #             alpha=0.9,  # 半透明，避免完全遮挡绿色谱线
                    #             edgecolors='red',  # 红色描边，增强视觉区分度
                    #             linewidths=0.8,  # 描边宽度
                    #             zorder=10)  # 置于顶层，确保不被其他元素遮挡

                q_point = max(0, abs_x - 15)  # Q点X坐标（全局）
                q_roi_x = q_point - x_sl  # Q点X坐标（ROI内）
                if 0 <= q_roi_x < w_sl:  # 确保Q点在ROI范围内
                    # 提取Q点位置的绿色通道（与R波逻辑相同）
                    q_green_channel = spectral_roi[:, q_roi_x, 1]
                    # 找到Q点位置频谱线的Y坐标（绿色值最大的像素，参考R波逻辑）
                    q_roi_y = np.argmax(q_green_channel)
                    q_global_y = y_sl + q_roi_y  # Q点Y坐标（全局，使用与R波相同的偏移-9）

                    # 绘制Q点黄色亮点（与Q点竖线颜色一致）
                    # plt.scatter(q_point, q_global_y,
                    #             color='yellow',  # 黄色填充（匹配Q点竖线颜色）
                    #             s=120,  # 比R波略小，避免遮挡
                    #             alpha=0.9,
                    #             edgecolors='yellow',  # 黑色描边，区分于R波
                    #             linewidths=0.6,
                    #             zorder=10)  # 顶层显示
                s_point = min(abs_x + 15, img_rgb.shape[1] - 1)  # S点X坐标（全局）
                s_roi_x = s_point - x_sl  # S点X坐标（ROI内）
                if 0 <= s_roi_x < w_sl:  # 确保S点在ROI范围内
                    s_green_channel = spectral_roi[:, s_roi_x, 1]
                    s_roi_y = np.argmax(s_green_channel)
                    s_global_y = y_sl + s_roi_y  # S点Y坐标（全局，使用与R波相同的偏移-9）
                    # plt.scatter(s_point, s_global_y,
                    #             color='blue',  # 蓝色填充（匹配S点竖线颜色）
                    #             s=120,  # 比R波略小，避免遮挡
                    #             alpha=0.9,
                    #             edgecolors='blue',  # 黑色描边，区分于R波
                    #             linewidths=0.6,
                    #             zorder=10)  # 顶层显示
                # first_group_x_threshold = 100  # 第一组Q点X坐标阈值（需根据实际图片调整，确保仅最左侧Q点满足）处理0609\0150图片
                # shift_pixels = 15  # 下移像素数（正值向下）
                # q_point = max(0, abs_x - 15)  # Q点X坐标（全局）
                # q_roi_x = q_point - x_sl
                # if 0 <= q_roi_x < w_sl:
                #     q_green_channel = spectral_roi[:, q_roi_x, 1]
                #     q_roi_y = np.argmax(q_green_channel)
                #     q_global_y = y_sl + q_roi_y  # 原始Y坐标
                #
                #     # 仅当Q点X坐标小于阈值（第一组）时下移
                #     if q_point < first_group_x_threshold:  # 核心条件：通过X坐标锁定第一组
                #         q_global_y += shift_pixels  # 第一组Q点下移
                #
                #     plt.scatter(q_point, q_global_y,
                #                 color='yellow', s=120, alpha=0.9,
                #                 edgecolors='yellow', linewidths=0.6, zorder=10)
                # s_point = min(abs_x + 15, img_rgb.shape[1] - 1)  # S点X坐标（全局）
                # s_roi_x = s_point - x_sl
                # if 0 <= s_roi_x < w_sl:
                #     s_green_channel = spectral_roi[:, s_roi_x, 1]
                #     s_roi_y = np.argmax(s_green_channel)
                #     s_global_y = y_sl + s_roi_y  # 原始Y坐标
                #     if s_point < (first_group_x_threshold + 20):  # S点X比Q点大，阈值+20确保包含第一组S点
                #         s_global_y += shift_pixels  # 第一组S点下移
                #
                #     plt.scatter(s_point, s_global_y,
                #                 color='blue', s=120, alpha=0.9,
                #                 edgecolors='blue', linewidths=0.6, zorder=10)

                #绘制Q点竖线（黄色虚线）
                # plt.axvline(x=q_point, color='yellow', linestyle='--', linewidth=1.2, alpha=0.7,
                #             label='Q点' if peak == peaks[0] else None)

                # 绘制S点竖线（蓝色虚线）
                # plt.axvline(x=s_point, color='blue', linestyle='--', linewidth=1.2, alpha=0.7,
                #             label='S点' if peak == peaks[0] else None)
            # 新增：绘制第二个下坡点
            for down_point in second_downslopes:
                abs_x =  down_point
                # plt.axvline(x=abs_x, color='cyan', linestyle='--', linewidth=1.5, alpha=0.8,
                #             label='第二下坡点' if down_point == second_downslopes[0] else "")
                roi_x = abs_x - x_sl  # 全局X转ROI局部X
                if 0 <= roi_x < w_sl:
                    green_channel = spectral_roi[:, roi_x, 1]
                    down_roi_y = np.argmax(green_channel)  # 取同列最高亮度点
                    down_global_y = y_sl + down_roi_y
                    # plt.scatter(abs_x, down_global_y,
                    #             color='cyan',  # 青色填充
                    #             s=140,
                    #             alpha=0.9,
                    #             edgecolors='blue',  # 蓝色描边
                    #             linewidths=0.8,
                    #             zorder=10)  # 确保在顶层

        # time_info, time_coords = self.extract_time_info(img_rgb)
        # x1, y1, x2, y2 = time_coords
        #
        # # 绘制时间区域框（绿色矩形）
        # time_rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
        #                           linewidth=2, edgecolor='lime',
        #                           facecolor='none', linestyle='-',
        #                           alpha=0.7, label='时间区域')
        # plt.gca().add_patch(time_rect)
        #
        # # 在框上方添加时间文本标注
        # plt.text((x1 + x2) // 2, y1 - 5, f'时间: {time_info}',
        #          color='lime', fontsize=12, ha='center',
        #          bbox=dict(facecolor='black', alpha=0.7))
        pixel_per_100ms, total_ticks, first_pair = self.detect_time_scale(img_rgb)
        if first_pair:
            x1, x2 = first_pair
            # 计算在原图中的y坐标（底部区域）
            y_bottom = height - 20
            y_top = height - 5

            # 绘制相邻竖线框
            # plt.plot([x1, x1], [y_bottom, y_top], 'm-', linewidth=3, alpha=0.7, label='时间刻度线')
            # plt.plot([x2, x2], [y_bottom, y_top], 'm-', linewidth=3, alpha=0.7)
            #
            # # 在两条线之间添加标注
            # mid_x = (x1 + x2) // 2
            # plt.text(mid_x, y_top - 10, '100ms',
            #          color='white', fontsize=10, ha='center',
            #          bbox=dict(facecolor='purple', alpha=0.7))

        # if peak_speed is not None and mean_speed_in_peak is not None:
        #     plt.annotate(
        #         f'峰值速度: {peak_speed:.1f} cm/s\n均值速度: {mean_speed_in_peak:.1f} cm/s',
        #         xy=(peak_x, peak_global_y),
        #         xytext=(peak_x + 50, peak_global_y - 50),
        #         fontsize=12,
        #         color='white',
        #         bbox=dict(facecolor='red', alpha=0.8),
        #         arrowprops=dict(facecolor='yellow', shrink=0.05)
        #     )
            print(f"\n=== 血流动力学参数 ===")
            print(f"峰值速度: {peak_speed:.1f} cm/s")
            print(f"平均速度: {mean_speed_in_peak:.1f} cm/s")
            if max_pressure_gradient is not None and mean_pressure_gradient is not None:
                print(f"最大压力梯度: {max_pressure_gradient:.2f} mmHg")
                print(f"平均压力梯度: {mean_pressure_gradient:.2f} mmHg")
        #     # 绘制零速度线
        #     plt.axhline(y=zero_line_global, color='r', linestyle='--', linewidth=2, alpha=0.7, label="零速度线")
        #     # 绘制包络线
            x_vals = np.linspace(x, x + w - 1, len(upper_envelope))
            valid_mask = ~np.isnan(upper_envelope) & ~np.isnan(lower_envelope)
            # plt.plot(x_vals[valid_mask], (y + upper_envelope)[valid_mask], 'r-', lw=2.5, alpha=0.9)  # label='上包络线'
            # plt.plot(x_vals[valid_mask], (y + lower_envelope)[valid_mask], 'b-', lw=2.5, alpha=0.9)  # label='下包络线'
        #
        #     for (x_end, y_end, typ) in concave_endpoints:
        #         if typ == 'right':
        #             plt.scatter(x_end, y_end, color='green', s=200, marker='o', edgecolor='black', linewidth=2)

                # else:
                #     plt.scatter(x_end, y_end, color='red', s=200, marker='^', edgecolor='black', linewidth=2)
            # if peak_envelope_x is not None and peak_envelope_y is not None:
            #     plt.plot(peak_envelope_x, peak_envelope_y, color='yellow', linewidth=1,
            #              path_effects=[pe.Stroke(linewidth=1, foreground='black'), pe.Normal()],
            #              alpha=0.9, label='峰值波形包络线')
            # if smoothed_x is not None and smoothed_y is not None:
            #     plt.plot(
            #         smoothed_x, smoothed_y,
            #         color='yellow', linewidth=1,
            #         path_effects=[pe.Stroke(linewidth=6, foreground='black'), pe.Normal()],
            #         alpha=0.9, label='平滑峰值波形包络线'
            #     )
            # if current_smoothed_x is not None and current_smoothed_y is not None:
            #     plt.plot(
            #         current_smoothed_x, current_smoothed_y,
            #         color='cyan',  # 青色曲线区分二次平滑
            #         linewidth=1.5,
            #         path_effects=[pe.Stroke(linewidth=4, foreground='blue'), pe.Normal()],  # 蓝边效果增强区分度
            #         alpha=0.8
            #     )# label='二次平滑峰值波形包络线'


                # left_bound_x = min(peak_envelope_x)
                # right_bound_x = max(peak_envelope_x)
                #
                # # 获取左右边界对应的y坐标（包络线上的点）
                # left_bound_y = peak_envelope_y[np.argmin(peak_envelope_x)]
                # right_bound_y = peak_envelope_y[np.argmax(peak_envelope_x)]
                #
                # # 绘制从左边界到零速度线的垂直线
                # plt.plot([left_bound_x, left_bound_x], [left_bound_y, zero_line_global],
                #          color='lime', linestyle='--', linewidth=2, alpha=0.7,
                #          label='临界边界线')
                #
                # # 绘制从右边界到零速度线的垂直线
                # plt.plot([right_bound_x, right_bound_x], [right_bound_y, zero_line_global],
                #          color='lime', linestyle='--', linewidth=2, alpha=0.7)
            if vti is not None:
                    print(f"速度时间积分(VTI): {vti:.2f} cm")
                    print(f"波形时长: {time_interval:.3f} 秒")


        #绘制速度刻度区域框（青色虚线框）
        if speed_scale_coords:
            x1, y1, w1, h1 = speed_scale_coords
            rect = plt.Rectangle((x1, y1), w1, h1, linewidth=3, edgecolor='cyan', facecolor='none',
                                 linestyle='--', alpha=0.9)#, label='速度刻度区域'
            plt.gca().add_patch(rect)
        # # 绘制峰值标记和速度信息
        # plt.axhline(y=peak_global_y, color='y', linestyle='--', linewidth=2, alpha=0.7)
        # plt.axhline(y=peak_global_y, color='y', linestyle='--', linewidth=2, alpha=0.7)#label="峰值位置"


        # 标题和图例
        title_str = "DSE-AutoSTV"#多普勒频谱分析与峰值定位
        # plt.title(title_str, pad=20, fontsize=15)

        handles, labels = plt.gca().get_legend_handles_labels()
        unique_labels = []
        unique_handles = []
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_labels.append(label)
                unique_handles.append(handle)
        #不删 plt.legend(unique_handles, unique_labels, loc='upper right', fontsize=10)
        plt.axis('off')
        plt.tight_layout()
        savepath=r"D:\桌面文件\3.png"
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        plt.show()
        #绘制子图
        # plt.figure(figsize=(12, 6))
        # plt.imshow(spectrum_roi, cmap='gray')
        # x_vals_roi = np.linspace(0, w - 1, len(upper_envelope))
        # valid_mask = ~np.isnan(upper_envelope) & ~np.isnan(lower_envelope)
        # plt.plot(x_vals_roi[valid_mask], upper_envelope[valid_mask], 'r-', lw=2.5, alpha=0.9, label='上包络线')
        # plt.plot(x_vals_roi[valid_mask], lower_envelope[valid_mask], 'b-', lw=2.5, alpha=0.9, label='下包络线')
        # if current_smoothed_x is not None and current_smoothed_y is not None:
        #     roi_smoothed_x = [px - x for px in current_smoothed_x]  # 局部X = 全局X - 频谱ROI左上角X
        #     roi_smoothed_y = [py - y for py in current_smoothed_y]  # 局部Y = 全局Y - 频谱ROI左上角Y
        #     valid_indices = [i for i in range(len(roi_smoothed_x))
        #                      if 0 <= roi_smoothed_x[i] < w and 0 <= roi_smoothed_y[i] < h_roi]
        #     filtered_x = [roi_smoothed_x[i] for i in valid_indices]
        #     filtered_y = [roi_smoothed_y[i] for i in valid_indices]
        #     plt.plot(
        #         filtered_x, filtered_y,
        #         color='cyan',
        #         linewidth=1.5,
        #         path_effects=[pe.Stroke(linewidth=4, foreground='blue'), pe.Normal()],  # 蓝边增强区分度
        #         alpha=0.8,
        #         label='二次平滑峰值波形包络线'
        #     )
        # if 'peaks' in locals() and isinstance(peaks, (list, np.ndarray)) and len(peaks) > 0:    # 确保peaks有效（Q点基于peaks计算）
        #     for peak_idx, abs_x in enumerate(peaks):
        #         q_point = max(0, abs_x - 15)  # Q点全局X坐标（原始逻辑）
        #         q_roi_x = q_point - x  # 频谱ROI左上角X为x，局部X = 全局X - x
        #         if 0 <= q_roi_x < w:  # 确保Q点在频谱ROI范围内
        #             # 绘制垂直虚线（Y范围覆盖整个频谱图高度）
        #             plt.axvline(
        #                 x=q_roi_x,  # 局部X坐标
        #                 color='yellow',
        #                 linestyle='--',
        #                 linewidth=1.2,
        #                 alpha=0.7,
        #                 label='Q点' if peak_idx == 0 else ""  # 仅第一个Q点显示图例
        #             )
        # # 4. 绘制第二下坡点虚线（青色虚线，与原图样式一致）
        # if 'second_downslopes' in locals() and second_downslopes:  # 确保second_downslopes有效
        #     for down_idx, down_point in enumerate(second_downslopes):
        #         abs_x = down_point  # 第二下坡点全局X坐标
        #         # 将全局X坐标转换为频谱ROI局部X坐标
        #         down_roi_x = abs_x - x  # 频谱ROI左上角X为x，局部X = 全局X - x
        #         if 0 <= down_roi_x < w:  # 确保下坡点在频谱ROI范围内
        #             # 绘制垂直虚线（Y范围覆盖整个频谱图高度）
        #             plt.axvline(
        #                 x=down_roi_x,  # 局部X坐标
        #                 color='cyan',
        #                 linestyle='--',
        #                 linewidth=1.5,
        #                 alpha=0.8,
        #                 label='第二下坡点' if down_idx == 0 else ""  # 仅第一个点显示图例
        #             )
        # plt.axis('off')
        # save_path = r"D:\桌面文件\1-贝塞尔曲线.png"
        # plt.savefig(save_path, dpi=300,bbox_inches='tight')
        # print(f"频谱分析结果已保存到: {save_path}")

        # ======================== 精准修复：仅替换黑色背景为白色，保留原始绿色线 ========================
        # if 'spectral_region' in locals() and spectral_region:
        #     x_sl, y_sl, w_sl, h_sl = spectral_region
        #     # 1. 裁剪绿色频谱线区域ROI（原始图像）
        #     green_line_roi = img_rgb[y_sl-5:y_sl+5 + h_sl, x_sl:x_sl + w_sl].copy()
        #     h, w = green_line_roi.shape[:2]
        #
        #     # 2. 定义“纯黑色背景”阈值（严格区分背景和线条）
        #     # 背景条件：RGB三通道均极低（接近纯黑），且绿色通道无明显信号
        #     bg_threshold = 20  # 背景阈值（设为20，比之前更低，避免误判绿色线）
        #     background_mask = (
        #             (green_line_roi[:, :, 0] < bg_threshold) &  # R < 20
        #             (green_line_roi[:, :, 1] < bg_threshold * 2) &  # G < 40（允许绿色线区域G稍高）
        #             (green_line_roi[:, :, 2] < bg_threshold)  # B < 20
        #     )
        #
        #     # 3. 定义“绿色线条”区域（排除背景，保留所有非黑色内容）
        #     # 线条区域 = 非背景区域（即：不是纯黑色的区域）
        #     line_mask = ~background_mask  # 取反：非背景即为线条/内容
        #     # 4. 创建结果图像：背景→白色，线条保留原始颜色
        #     result_image = np.ones_like(green_line_roi) * 255  # 白色背景
        #     result_image[line_mask] = green_line_roi[line_mask]  # 非背景区域保留原始颜色
        #     # 5. 显示结果（调整高度，避免标题重叠）
        #     plt.figure(figsize=(15, 4))  # 宽15，高4
        #     plt.imshow(result_image)
        #
        #     for idx, peak in enumerate(peaks):
        #         abs_x = peak  # R波全局X坐标
        #
        #         # ---------------------- R波峰值点（红色） ----------------------
        #         roi_x = abs_x - x_sl  # 全局X → ROI局部X
        #         if 0 <= roi_x < w_sl:
        #             # 提取R波列绿色通道，找到峰值Y坐标（局部+全局转换）
        #             green_channel = spectral_roi[:, roi_x, 1]
        #             peak_roi_y = np.argmax(green_channel)
        #             peak_global_y_roi = y_sl + peak_roi_y -3  # 局部Y → 全局Y（原始偏移-9）
        #             # 转换为频谱线图像的局部坐标（用于在white_bg上绘制）
        #             peak_local_y = peak_global_y_roi - y_sl  # 全局Y → 频谱线图像局部Y
        #             plt.scatter(roi_x, peak_local_y,  # 使用ROI局部坐标（x=roi_x, y=peak_local_y）
        #                         color='red', s=140, alpha=0.9,
        #                         edgecolors='red', linewidths=0.8, zorder=10,
        #                         label='R波峰值' if idx == 0 else "")
        #
        #         # # ---------------------- Q点（黄色） ----------------------
        #         q_point = max(0, abs_x - 15)  # Q点全局X坐标
        #         q_roi_x = q_point - x_sl  # 全局X → ROI局部X
        #         if 0 <= q_roi_x < w_sl:
        #             q_green_channel = spectral_roi[:, q_roi_x, 1]
        #             q_roi_y = np.argmax(q_green_channel)
        #             q_global_y_roi = y_sl + q_roi_y +3 # 全局Y
        #             q_local_y = q_global_y_roi - y_sl  # 频谱线图像局部Y
        #             plt.scatter(q_roi_x, q_local_y,
        #                         color='yellow', s=120, alpha=0.9,
        #                         edgecolors='yellow', linewidths=0.6, zorder=10,
        #                         label='Q点' if idx == 0 else "")
        #
        #         # ---------------------- S点（蓝色） ----------------------
        #         s_point = min(abs_x + 15, img_rgb.shape[1] - 1)  # S点全局X坐标
        #         s_roi_x = s_point - x_sl  # 全局X → ROI局部X
        #         if 0 <= s_roi_x < w_sl:
        #             s_green_channel = spectral_roi[:, s_roi_x, 1]
        #             s_roi_y = np.argmax(s_green_channel)
        #             s_global_y_roi = y_sl + s_roi_y +3 # 全局Y
        #             s_local_y = s_global_y_roi - y_sl  # 频谱线图像局部Y
        #             plt.scatter(s_roi_x, s_local_y,
        #                         color='blue', s=120, alpha=0.9,
        #                         edgecolors='blue', linewidths=0.6, zorder=10,
        #                         label='S点' if idx == 0 else "")
        #
        #         # 2. 叠加第二下坡点（青色）
        #     for down_idx, down_point in enumerate(second_downslopes):
        #         abs_x = down_point  # 下坡点全局X坐标
        #         roi_x = abs_x - x_sl  # 全局X → ROI局部X
        #         if 0 <= roi_x < w_sl:
        #             green_channel = spectral_roi[:, roi_x, 1]
        #             down_roi_y = np.argmax(green_channel)
        #             down_global_y_roi = y_sl + down_roi_y  +3# 全局Y
        #             down_local_y = down_global_y_roi - y_sl  # 频谱线图像局部Y
        #             plt.scatter(roi_x, down_local_y,
        #                         color='cyan', s=140, alpha=0.9,
        #                         edgecolors='blue', linewidths=0.8, zorder=10,
        #                         label='第二下坡点' if down_idx == 0 else "")
        #     # plt.axis('off')
        #     # savepath=r"D:\桌面文件\3.png"
        #     # plt.savefig(savepath, dpi=300, bbox_inches='tight')
        #     plt.show()
        plt.show()
        print("峰值定位完成")
        return True

if __name__ == "__main__":
    analyzer = DopplerEnvelopeAnalyzer(
        prf=17,
        spectrum_region_ratio=(0.49, 1.0),  # 根据您的图像调整频谱区域（下半部分）
        baseline_exclude_height=40,
        text_region_width_ratio=0.1,
        left_text_width_ratio=0.14,
        left_text_height_ratio=0.66,
        speed_scale_height_ratio=(0.44, 0.955),
        speed_scale_width_ratio=0.07
    )

    # 请修改为实际图像路径
    image_path = r"D:\雨婷\超声多普勒分析\heatmap\val2017\50560820240722_LY-FUHE-KONGGE_20240722090753241.jpg"#"D:\桌面文件\jpg图片\精简\0409精简\IM_0358.jpg"

    print("⏳ 正在处理医学图像...")
    if analyzer.visualize(image_path):
        print("✅ 处理完成！")
    else:
        print("❌ 处理失败！")