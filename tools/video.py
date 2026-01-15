import base64
from io import BytesIO

import streamlit as st
import traceback
from turtle import width, pd
#处理一组图像，出折线图
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import pywt
import os
import cv2
import heartpy as hp
from scipy.interpolate import CubicSpline, make_interp_spline
from scipy.ndimage import binary_erosion, binary_dilation, binary_fill_holes, binary_closing, gaussian_filter1d
from scipy.signal import savgol_filter, find_peaks
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
        pass

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
        pass

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
    pass

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
        pass

    def extract_envelopes(self, spectrum_roi, smooth_method='lowess'):
        """改进的包络线提取方法 - 多种平滑方法"""
        try:
            h, w = spectrum_roi.shape
            upper_envelope = np.full(w, np.nan)
            lower_envelope = np.full(w, np.nan)
            valid_width = int(w * 0.995)
            if valid_width <= 0:
                return upper_envelope, lower_envelope  # 宽度无效时直接返回空包络
            total_groups = 0

            # 1. 预处理
            filtered_roi = cv2.GaussianBlur(spectrum_roi.astype(np.float32), (5, 5), 1.0)

            # 3. 动态阈值计算与二值化
            valid_roi = filtered_roi[:, :valid_width]
            global_median = np.median(valid_roi)
            global_std = np.std(valid_roi)
            global_max = np.max(valid_roi)
            dynamic_threshold = min(
                max(global_median + 0.5 * global_std, 0.1 * global_max),
                0.1 * global_max
            )
            thresholded = np.zeros_like(filtered_roi)
            thresholded[filtered_roi > dynamic_threshold] = 255

            # 4. 候选点分组（颜色区分）
            group_visualization = np.zeros((h, w, 3), dtype=np.uint8)  # 彩色分组图
            # 新增：存储原始包络点（未插值前）
            raw_upper_points = np.full(w, np.nan)
            raw_lower_points = np.full(w, np.nan)

            for x in range(valid_width):
                column = filtered_roi[:, x]
                candidate_pixels = np.where(column > dynamic_threshold)[0]
                # 备用阈值
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

            upper_envelope = raw_upper_points
            lower_envelope = raw_lower_points

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

            # 应用选择的平滑方法
            upper_envelope = smooth_envelope(upper_envelope, smooth_method)
            lower_envelope = smooth_envelope(lower_envelope, smooth_method)


            return upper_envelope, lower_envelope

        except Exception as e:
            print(f"包络提取错误: {str(e)}")
            return np.full(w, np.nan), np.full(w, np.nan)
        pass

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


            return speed_scale_roi, scale_left_col, scale_top, scale_bottom, scale_width

        except Exception as e:
            print(f"速度刻度提取错误: {str(e)}")
            return None, 0, 0, 0, 0
        pass

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
    pass

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
        pass

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
        pass
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
                return match.group(1)  # (x1, y1, x2, y2)
            else:
                time_pattern = r'(\d{2}:\d{2})'
                match = re.search(time_pattern, text)
                if match:
                    return match.group(1) + ":00", (x1, y1, x2, y2)
                return "未识别到时间信息", (x1, y1, x2, y2)
        except Exception as e:
            print(f"时间信息提取错误: {str(e)}")
            return "时间提取失败", (0, 0, 0, 0)
        pass
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
            return x, y - 1, w, h + 1

        except Exception as e:
            print(f"频谱线检测错误: {str(e)}")
            return None
        pass
    def analyze_spectral_peaks(self, img_rgb, spectral_region):
        """分析频谱峰值，返回全局坐标的峰值和相邻峰值前1/3位置的蓝色竖线"""
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
        pass
    def trace_peak_envelope(self, upper_envelope, spectrum_roi, peak_idx, img_rgb, x_offset, y_offset, peaks,
                            second_downslopes):
        """仅追踪上包络线（强制以left_bound/right_bound为交点并添加为右端点）"""
        try:
            h_roi, w_roi = spectrum_roi.shape[:2]
            if w_roi <= 0 or h_roi <= 0:
                return None, None, []

            roi_global_x_min = x_offset
            current_peak_roi = peak_idx  # 当前峰值在ROI内的x坐标（像素）
            roi_peaks = sorted(
                [p - roi_global_x_min for p in peaks if 0 <= (p - roi_global_x_min) < w_roi])  # 红色竖线（ROI内，升序）
            roi_second_downslopes = sorted([s - roi_global_x_min for s in second_downslopes if
                                            0 <= (s - roi_global_x_min) < w_roi])  # 蓝色竖线（ROI内，升序）

            # ---------------------- 关键：计算全局标准间距（必须优先执行） ----------------------
            # 全局第一个蓝色竖线（从左到右第一个）
            first_blue = roi_second_downslopes[0] if roi_second_downslopes else None
            # 其右侧第一个红色竖线（严格匹配“蓝线→红线”的标准波形间距）
            right_red_of_blue = next((r for r in roi_peaks if r > first_blue), None) if first_blue is not None else None
            # 计算标准间距（蓝色竖线到红色竖线的像素距离）
            standard_spacing = right_red_of_blue - first_blue if (
                    first_blue is not None and right_red_of_blue is not None) else None

            # 若无法计算标准间距（无蓝线或蓝线右侧无红线），直接报错并返回
            if standard_spacing is None:
                print(f"错误：无法计算全局标准间距（蓝线={first_blue}, 红线={right_red_of_blue}）")
                return None, None, []

            # ---------------------- 峰值左侧线条分析 ----------------------
            left_reds = [r for r in roi_peaks if r < current_peak_roi]  # 峰值左侧红色竖线
            left_blues = [s for s in roi_second_downslopes if s < current_peak_roi]  # 峰值左侧蓝色竖线
            closest_red_left = max(left_reds) if left_reds else None  # 左侧最近红色竖线
            closest_blue_left = max(left_blues) if left_blues else None  # 左侧最近蓝色竖线

            # ---------------------- 特殊情况处理（统一使用标准间距） ----------------------
            # 情况1：峰值左侧无任何竖线（红/蓝均不存在）
            if closest_red_left is None and closest_blue_left is None:
                print(f"特殊情况：峰值左侧无竖线 → 使用标准间距={standard_spacing}px")
                left_bound = max(0, current_peak_roi - standard_spacing // 2)  # 峰值向左半间距
                right_bound = min(w_roi - 1, current_peak_roi + standard_spacing // 2)  # 峰值向右半间距

            # 情况2：峰值左侧最近为红色竖线（红色比蓝色更近）
            elif (
                    closest_red_left is not None and closest_blue_left is not None and closest_red_left > closest_blue_left):
                print(f"特殊情况：红色竖线更近 → 使用标准间距={standard_spacing}px")
                left_bound = max(0, current_peak_roi - standard_spacing // 2)  # 峰值向左半间距
                right_bound = min(w_roi - 1, current_peak_roi + standard_spacing // 2)  # 峰值向右半间距

            # 情况3：峰值左侧仅存在红色竖线（无蓝色竖线）
            elif closest_red_left is not None and closest_blue_left is None:
                print(f"特殊情况：仅存在红色竖线 → 使用标准间距={standard_spacing}px")
                left_bound = max(0, current_peak_roi - standard_spacing // 2)  # 峰值向左半间距
                right_bound = min(w_roi - 1, current_peak_roi + standard_spacing // 2)  # 峰值向右半间距

            # ---------------------- 正常情况：左侧最近为蓝色竖线 ----------------------
            else:
                # 左边界：左侧最近蓝色竖线（原有逻辑，确保与蓝线对齐）
                left_bound = closest_blue_left if closest_blue_left is not None else max(0, int(current_peak_roi * 0.9))
                # 右边界：左侧蓝线 + 标准间距（强制与标准间距一致）
                right_bound = min(w_roi - 1, left_bound + standard_spacing)  # 关键修复：蓝线+标准间距=红线位置

            # ---------------------- 验证间距一致性（调试用，可保留） ----------------------
            actual_spacing = right_bound - left_bound
            if abs(actual_spacing - standard_spacing) > 1:  # 允许1像素误差（整数除法导致）
                print(f"警告：实际间距({actual_spacing}px)与标准间距({standard_spacing}px)不一致！")

            # ---------------------- 包络线生成逻辑（保持不变） ----------------------
            velocity_threshold_y_roi = np.interp(10.0, self.speed_values, self.speed_positions)
            valid_columns = []
            for x in range(left_bound, right_bound + 1):
                if x >= len(upper_envelope) or np.isnan(upper_envelope[x]):
                    continue
                if upper_envelope[x] < velocity_threshold_y_roi:
                    valid_columns.append(x)
            for x in [left_bound, right_bound]:
                if x not in valid_columns and 0 <= x < len(upper_envelope) and not np.isnan(upper_envelope[x]):
                    valid_columns.append(x)
            valid_columns = sorted(valid_columns)
            if len(valid_columns) < 2:
                print(f"有效列不足（{len(valid_columns)}列）")
                return None, None, []

            x_coords = [x_offset + x for x in valid_columns]
            y_coords = [y_offset + upper_envelope[x] for x in valid_columns]
            blue_intersect = (x_offset + left_bound, y_offset + upper_envelope[left_bound]) if left_bound < len(
                upper_envelope) else (x_offset + left_bound, y_offset)
            red_intersect = (x_offset + right_bound, y_offset + upper_envelope[right_bound]) if right_bound < len(
                upper_envelope) else (x_offset + right_bound, y_offset)
            forced_right_endpoints = [
                (blue_intersect[0], blue_intersect[1], 'right'),
                (red_intersect[0], red_intersect[1], 'right')
            ]

            return np.array(x_coords), np.array(y_coords), forced_right_endpoints

        except Exception as e:
            print(f"包络线追踪错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None, []
        pass
    def detect_concave_endpoints(self, x_coords, y_coords, min_segment_length=3):
        """
        检测V/U形凹区间的两端端点（起点和终点）

        参数:
            x_coords: 包络线x坐标（全局）
            y_coords: 包络线y坐标（全局）
            min_segment_length: 最小凹区间长度（避免短噪声干扰，默认3个点）

        返回:
            端点列表：[(x, y, 'left'), (x, y, 'right'), ...]
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
    pass
    def smooth_connect_anchors(self, peak_envelope_x, peak_envelope_y, concave_endpoints):
        """
        修改版：在每对相邻的绿色圆形端点之间生成平滑曲线

        参数:
            peak_envelope_x: 原始包络线X坐标
            peak_envelope_y: 原始包络线Y坐标
            concave_endpoints: 凹区间端点列表 [(x, y, 'left'/'right'), ...]

        返回:
            修正后的包络线坐标 (new_x, new_y)
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
    pass

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

            if len(valid_x) < 3:  # 至少需要5个点构成连续波形
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
        pass
    def detect_time_scale(self, img_rgb):
        """改进的时间刻度检测方法（添加竖线间隔≥10像素限制）"""
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
            threshold = np.max(projection) * 0.1  # 更低的阈值

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

            # 5. 改进的间距计算（增加竖线间隔≥10像素限制）
            diffs = np.diff(x_positions)  # 相邻刻度线的间距（像素）
            median_diff = np.median(diffs)

            # 过滤异常间距（新增：必须 >10像素）
            valid_diffs = []
            valid_pairs = []
            for i in range(len(diffs)):
                # 同时满足：
                # 1. 0.5~2倍中位数间距（原逻辑，确保等距）
                # 2. 间距 >10像素（新增限制，避免密集干扰）
                if (0.5 * median_diff < diffs[i] < 2.0 * median_diff) and (diffs[i] > 30):
                    valid_diffs.append(diffs[i])
                    valid_pairs.append((x_positions[i], x_positions[i + 1]))

            if not valid_diffs:  # 无有效间距（可能因新增限制过滤掉所有间距）
                return None, None, None

            # 6. 计算平均间距
            px_per_100ms = int(np.mean(valid_diffs))

            # 7. 物理范围验证（可保留原逻辑）
            if not (5 <= px_per_100ms <= 150):
                return None, None, None

            # 8. 返回第一对有效竖线的位置和总刻度数
            first_pair = valid_pairs[0] if valid_pairs else None
            total_ticks = len(x_positions)

            return px_per_100ms, total_ticks, first_pair

        except Exception as e:
            print(f"时间刻度检测错误: {e}")
            return None, None, None
        pass

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
        pass
    def visualize(self, image_path):
        """可视化分析结果（包含峰值检测和速度刻度线框选）并计算峰值速度"""
        global peak_speed, mean_speed_in_peak, zero_line_global, max_pressure_gradient, mean_pressure_gradient, peak_envelope_x, peak_envelope_y, x_sl, vti, fig1
        img_gray, img_rgb = self.load_image(image_path, return_original=True)
        if img_gray is None or img_rgb is None:
            return False
        time_info = self.extract_time_info(img_rgb)
        print(f"图像时间: {time_info}")
        # 获取图像尺寸
        height, width = img_gray.shape[:2]
        y_start = int(height * 0.08)  # 10%高度
        y_end = int(height * 0.4)  # 40%高度
        x_start = int(width * 0.2)  # 20%宽度
        x_end = int(width * 0.8)  # 80%宽度

        # 绘制矩形框e
        # cv2.rectangle(img_rgb,
        #               (x_start, y_start),
        #               (x_end, y_end),
        #               color=(255, 0, 255),  # 品红色
        #               thickness=3)
        # 定位频谱区域
        x, y, w, h_roi = self.locate_spectrum(img_gray)
        spectrum_roi = img_gray[y:y + h_roi, x:x + w]
        # cv2.rectangle(img_rgb,
        #               (x, y),
        #               (x + w, y + h_roi),
        #               (0, 0, 255),  # 红色框
        #               2)
        upper_envelope, lower_envelope = self.extract_envelopes(spectrum_roi,smooth_method='moving_avg')

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
            peaks, smoothed, second_downslopes = self.analyze_spectral_peaks(img_rgb, spectral_region)
        else:
            peaks = []
            second_downslopes = []

        # 提取速度刻度区域并识别速度值
        speed_scale_roi, scale_left_col, scale_top, scale_bottom, scale_width = self.extract_speed_scale_region(img_rgb)
        speed_scale_coords = None  # 存储速度刻度区域坐标

        if speed_scale_roi is not None:
            peak_y_relative = peak_global_y - scale_top  # 相对于刻度区域顶部的位置
            peak_speed = self.get_speed_from_relative_position(peak_y_relative)

            zero_line_global = y + self.detect_zero_velocity_line(img_rgb[y:y + h_roi, x:x + w])
            peaks_left_shifted = np.clip(peaks - 15, 0, img_rgb.shape[1] - 1)
            peak_envelope_x, peak_envelope_y, right_endpoints= self.trace_peak_envelope(
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
                    min_segment_length=3 # 保持与原方法一致的阈值
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
            mean_speed_in_peak=self.calculate_average_velocity(current_smoothed_x, current_smoothed_y,img_rgb,scale_top)
            max_pressure_gradient, mean_pressure_gradient = self.calculate_pressure_gradients(peak_speed,
                                                                                              mean_speed_in_peak)
            vti, time_interval = self.calculate_velocity_time_integral(
                current_smoothed_x, current_smoothed_y, img_rgb, scale_top
            )

        fig_with_rect = img_rgb.copy()
        cv2.rectangle(fig_with_rect,
                      (x_sl, y_sl),
                      (x_sl + w_sl, y_sl + h_sl),
                      color=(0, 255, 0),  # 绿色
                      thickness=2)
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.imshow(fig_with_rect)
        if spectral_region and len(peaks) > 0:
            for peak in peaks:
                abs_x = peak
                # plt.axvline(x=abs_x-15, color='red', linestyle='-', linewidth=1.5, alpha=0.8,
                #             label='有效峰值' if peak == peaks[0] else "")
                roi_x = abs_x - x_sl  # 将全局X坐标转换为ROI内的局部X坐标
                if 0 <= roi_x < w_sl:  # 确保X坐标在ROI范围内
                    green_channel = spectral_roi[:, roi_x, 1]  # 提取ROI中第roi_x列的绿色通道值（0~255）
                    peak_roi_y = np.argmax(green_channel)  # 获取绿色值最大的像素行索引（ROI内局部坐标）
                    peak_global_y1 = y_sl + peak_roi_y- 9  # 频谱区域起点Y + ROI内Y坐标
                    ax.scatter(abs_x, peak_global_y1,
                                color='red',  # 黄色填充
                                s=140,  # 点大小（可根据图像分辨率调整）
                                alpha=0.9,  # 半透明，避免完全遮挡绿色谱线
                                edgecolors='red',  # 红色描边，增强视觉区分度
                                linewidths=0.8,  # 描边宽度
                                zorder=10)  # 置于顶层，确保不被其他元素遮挡
                q_point = max(0, abs_x - 15)  # Q点X坐标（全局）
                q_roi_x = q_point - x_sl  # Q点X坐标（ROI内）
                if 0 <= q_roi_x < w_sl:  # 确保Q点在ROI范围内
                    # 提取Q点位置的绿色通道（与R波逻辑相同）
                    q_green_channel = spectral_roi[:, q_roi_x, 1]
                    # 找到Q点位置频谱线的Y坐标（绿色值最大的像素，参考R波逻辑）
                    q_roi_y = np.argmax(q_green_channel)
                    q_global_y = y_sl + q_roi_y  # Q点Y坐标（全局，使用与R波相同的偏移-9）

                    # 绘制Q点黄色亮点（与Q点竖线颜色一致）
                    ax.scatter(q_point, q_global_y,
                                color='yellow',  # 黄色填充（匹配Q点竖线颜色）
                                s=120,  # 比R波略小，避免遮挡
                                alpha=0.9,
                                edgecolors='yellow',  # 黑色描边，区分于R波
                                linewidths=0.6,
                                zorder=10)  # 顶层显示
                s_point = min(abs_x + 15, img_rgb.shape[1] - 1)  # S点X坐标（全局）
                s_roi_x = s_point - x_sl  # S点X坐标（ROI内）
                if 0 <= s_roi_x < w_sl:  # 确保S点在ROI范围内
                    s_green_channel = spectral_roi[:, s_roi_x, 1]
                    s_roi_y = np.argmax(s_green_channel)
                    s_global_y = y_sl + s_roi_y  # S点Y坐标（全局，使用与R波相同的偏移-9）
                    ax.scatter(s_point, s_global_y,
                                color='blue',  # 蓝色填充（匹配S点竖线颜色）
                                s=120,  # 比R波略小，避免遮挡
                                alpha=0.9,
                                edgecolors='blue',  # 黑色描边，区分于R波
                                linewidths=0.6,
                                zorder=10)  # 顶层显示
                # 绘制Q点竖线（黄色虚线）
                ax.axvline(x=q_point, color='yellow', linestyle='--', linewidth=1.2, alpha=0.7,
                            label='Q点' if peak == peaks[0] else "")
            # 新增：绘制第二个下坡点
            for down_point in second_downslopes:
                abs_x =  down_point
                ax.axvline(x=abs_x, color='cyan', linestyle='--', linewidth=1.5, alpha=0.8,
                            label='第二下坡点' if down_point == second_downslopes[0] else "")
                roi_x = abs_x - x_sl  # 全局X转ROI局部X
                if 0 <= roi_x < w_sl:
                    green_channel = spectral_roi[:, roi_x, 1]
                    down_roi_y = np.argmax(green_channel)  # 取同列最高亮度点
                    down_global_y = y_sl + down_roi_y
                    plt.scatter(abs_x, down_global_y,
                                color='cyan',  # 青色填充
                                s=140,
                                alpha=0.9,
                                edgecolors='blue',  # 蓝色描边
                                linewidths=0.8,
                                zorder=10)  # 确保在顶层
            title_str = "DSE-AutoSTV"  # "多普勒频谱分析与峰值定位"
            plt.title(title_str, pad=20, fontsize=15)
            handles, labels = plt.gca().get_legend_handles_labels()
            unique_labels = []
            unique_handles = []
            for handle, label in zip(handles, labels):
                if label not in unique_labels:
                    unique_labels.append(label)
                    unique_handles.append(handle)
            ax.axis('off')


            fig1, ax1 = plt.subplots(figsize=(15, 8))
            ax1.imshow(img_rgb)
            x_vals = np.linspace(x, x + w - 1, len(upper_envelope))
            valid_mask = ~np.isnan(upper_envelope) & ~np.isnan(lower_envelope)
            ax1.plot(x_vals[valid_mask], (y + upper_envelope)[valid_mask], 'r-', lw=2.5, alpha=0.9)  # label='上包络线'
            ax1.plot(x_vals[valid_mask], (y + lower_envelope)[valid_mask], 'b-', lw=2.5, alpha=0.9)  # label='下包络线'
            title_str = "DSE-AutoSTV"  # "多普勒频谱分析与峰值定位"
            plt.title(title_str, pad=20, fontsize=15)
            ax1.set_xticks([])  # 移除X轴刻度
            ax1.set_yticks([])  # 移除Y轴刻度
            ax1.spines['top'].set_visible(False)  # 隐藏上边框
            ax1.spines['bottom'].set_visible(False)  # 隐藏下边框
            ax1.spines['left'].set_visible(False)  # 隐藏左边框
            ax1.spines['right'].set_visible(False)  # 隐藏右边框
            plt.tight_layout()

            fig2, ax2 = plt.subplots(figsize=(15, 8))
            ax2.imshow(img_rgb)
            if spectral_region and len(peaks) > 0:
                for peak in peaks:
                    abs_x = peak
                    q_point = max(0, abs_x - 15)  # Q点X坐标（全局）
                    q_roi_x = q_point - x_sl  # Q点X坐标（ROI内）
                    if 0 <= q_roi_x < w_sl:  # 确保Q点在ROI范围内
                        # 提取Q点位置的绿色通道（与R波逻辑相同）
                        q_green_channel = spectral_roi[:, q_roi_x, 1]
                        # 找到Q点位置频谱线的Y坐标（绿色值最大的像素，参考R波逻辑）
                        q_roi_y = np.argmax(q_green_channel)
                        q_global_y = y_sl + q_roi_y  # Q点Y坐标（全局，使用与R波相同的偏移-9）
                    s_point = min(abs_x + 15, img_rgb.shape[1] - 1)  # S点X坐标（全局）
                    s_roi_x = s_point - x_sl  # S点X坐标（ROI内）
                    if 0 <= s_roi_x < w_sl:  # 确保S点在ROI范围内
                        s_green_channel = spectral_roi[:, s_roi_x, 1]
                        s_roi_y = np.argmax(s_green_channel)
                        s_global_y = y_sl + s_roi_y  # S点Y坐标（全局，使用与R波相同的偏移-9）
                    # 绘制Q点竖线（黄色虚线）
                    ax2.axvline(x=q_point, color='yellow', linestyle='--', linewidth=1.2, alpha=0.7,
                               label='Q点' if peak == peaks[0] else "")
                # 新增：绘制第二个下坡点
                for down_point in second_downslopes:
                    abs_x = down_point
                    ax2.axvline(x=abs_x, color='cyan', linestyle='--', linewidth=1.5, alpha=0.8,
                               label='第二下坡点' if down_point == second_downslopes[0] else "")
            if peak_speed is not None and mean_speed_in_peak is not None:
                ax2.annotate(
                    f'Peak',
                    xy=(peak_x, peak_global_y),
                    xytext=(peak_x + 50, peak_global_y - 50),
                    fontsize=12,
                    color='white',
                    bbox=dict(facecolor='red', alpha=0.8),
                    arrowprops=dict(facecolor='yellow', shrink=0.05)
                )
            ax2.axhline(y=peak_global_y, color='y', linestyle='--', linewidth=2, alpha=0.7)
            x_vals = np.linspace(x, x + w - 1, len(upper_envelope))
            valid_mask = ~np.isnan(upper_envelope) & ~np.isnan(lower_envelope)
            ax2.plot(x_vals[valid_mask], (y + upper_envelope)[valid_mask], 'r-', lw=2.5, alpha=0.9)  # label='上包络线'
            ax2.plot(x_vals[valid_mask], (y + lower_envelope)[valid_mask], 'b-', lw=2.5, alpha=0.9)  # label='下包络

            title_str = "DSE-AutoSTV"  # "多普勒频谱分析与峰值定位"
            plt.title(title_str, pad=20, fontsize=15)
            ax2.set_xticks([])  # 移除X轴刻度
            ax2.set_yticks([])  # 移除Y轴刻度
            ax2.spines['top'].set_visible(False)  # 隐藏上边框
            ax2.spines['bottom'].set_visible(False)  # 隐藏下边框
            ax2.spines['left'].set_visible(False)  # 隐藏左边框
            ax2.spines['right'].set_visible(False)  # 隐藏右边框
            plt.tight_layout()

            fig2, ax2 = plt.subplots(figsize=(15, 8))
            ax2.imshow(img_rgb)
            if peak_speed is not None and mean_speed_in_peak is not None:
                ax2.annotate(
                    f'Peak',
                    xy=(peak_x, peak_global_y),
                    xytext=(peak_x + 50, peak_global_y - 50),
                    fontsize=12,
                    color='white',
                    bbox=dict(facecolor='red', alpha=0.8),
                    arrowprops=dict(facecolor='yellow', shrink=0.05)
                )
            ax2.axhline(y=peak_global_y, color='y', linestyle='--', linewidth=2, alpha=0.7)
            x_vals = np.linspace(x, x + w - 1, len(upper_envelope))
            valid_mask = ~np.isnan(upper_envelope) & ~np.isnan(lower_envelope)
            ax2.plot(x_vals[valid_mask], (y + upper_envelope)[valid_mask], 'r-', lw=2.5, alpha=0.9)  # label='上包络线'
            ax2.plot(x_vals[valid_mask], (y + lower_envelope)[valid_mask], 'b-', lw=2.5, alpha=0.9)  # label='下包络
            title_str = "DSE-AutoSTV"  # "多普勒频谱分析与峰值定位"
            plt.title(title_str, pad=20, fontsize=15)
            ax2.set_xticks([])  # 移除X轴刻度
            ax2.set_yticks([])  # 移除Y轴刻度
            ax2.spines['top'].set_visible(False)  # 隐藏上边框
            ax2.spines['bottom'].set_visible(False)  # 隐藏下边框
            ax2.spines['left'].set_visible(False)  # 隐藏左边框
            ax2.spines['right'].set_visible(False)  # 隐藏右边框
            plt.tight_layout()

            fig3,ax3=plt.subplots(figsize=(15, 8))
            ax3.imshow(img_rgb)
            if spectral_region and len(peaks) > 0:
                for peak in peaks:
                    abs_x = peak
                    q_point = max(0, abs_x - 15)  # Q点X坐标（全局）
                    q_roi_x = q_point - x_sl  # Q点X坐标（ROI内）
                    if 0 <= q_roi_x < w_sl:  # 确保Q点在ROI范围内
                        # 提取Q点位置的绿色通道（与R波逻辑相同）
                        q_green_channel = spectral_roi[:, q_roi_x, 1]
                        # 找到Q点位置频谱线的Y坐标（绿色值最大的像素，参考R波逻辑）
                        q_roi_y = np.argmax(q_green_channel)
                        q_global_y = y_sl + q_roi_y  # Q点Y坐标（全局，使用与R波相同的偏移-9）
                    s_point = min(abs_x + 15, img_rgb.shape[1] - 1)  # S点X坐标（全局）
                    s_roi_x = s_point - x_sl  # S点X坐标（ROI内）
                    if 0 <= s_roi_x < w_sl:  # 确保S点在ROI范围内
                        s_green_channel = spectral_roi[:, s_roi_x, 1]
                        s_roi_y = np.argmax(s_green_channel)
                        s_global_y = y_sl + s_roi_y  # S点Y坐标（全局，使用与R波相同的偏移-9）
                    # 绘制Q点竖线（黄色虚线）
                    ax3.axvline(x=q_point, color='yellow', linestyle='--', linewidth=1.2, alpha=0.7,
                                label='Q点' if peak == peaks[0] else "")
                # 新增：绘制第二个下坡点
                for down_point in second_downslopes:
                    abs_x = down_point
                    ax3.axvline(x=abs_x, color='cyan', linestyle='--', linewidth=1.5, alpha=0.8,
                                label='第二下坡点' if down_point == second_downslopes[0] else "")
            if current_smoothed_x is not None and current_smoothed_y is not None:
                ax3.plot(
                    current_smoothed_x, current_smoothed_y,
                    color='cyan',  # 青色曲线区分二次平滑
                    linewidth=1.5,
                    path_effects=[pe.Stroke(linewidth=4, foreground='blue'), pe.Normal()],  # 蓝边效果增强区分度
                    alpha=0.8,
                    label='二次平滑峰值波形包络线'
                )
            ax3.axhline(y=peak_global_y, color='y', linestyle='--', linewidth=2, alpha=0.7)
            x_vals = np.linspace(x, x + w - 1, len(upper_envelope))
            valid_mask = ~np.isnan(upper_envelope) & ~np.isnan(lower_envelope)
            ax3.plot(x_vals[valid_mask], (y + upper_envelope)[valid_mask], 'r-', lw=2.5, alpha=0.9)  # label='上包络线'
            ax3.plot(x_vals[valid_mask], (y + lower_envelope)[valid_mask], 'b-', lw=2.5, alpha=0.9)  # label='下包络
            # title_str = "DSE-AutoSTV"  # "多普勒频谱分析与峰值定位"
            # plt.title(title_str, pad=20, fontsize=15)
            ax3.set_xticks([])  # 移除X轴刻度
            ax3.set_yticks([])  # 移除Y轴刻度
            ax3.spines['top'].set_visible(False)  # 隐藏上边框
            ax3.spines['bottom'].set_visible(False)  # 隐藏下边框
            ax3.spines['left'].set_visible(False)  # 隐藏左边框
            ax3.spines['right'].set_visible(False)  # 隐藏右边框
            plt.tight_layout()

            fig4, ax4 = plt.subplots(figsize=(15, 8))
            ax4.imshow(img_rgb)
            if spectral_region and len(peaks) > 0:
                for peak in peaks:
                    abs_x = peak
                    q_point = max(0, abs_x - 15)  # Q点X坐标（全局）
                    q_roi_x = q_point - x_sl  # Q点X坐标（ROI内）
                    if 0 <= q_roi_x < w_sl:  # 确保Q点在ROI范围内
                        # 提取Q点位置的绿色通道（与R波逻辑相同）
                        q_green_channel = spectral_roi[:, q_roi_x, 1]
                        # 找到Q点位置频谱线的Y坐标（绿色值最大的像素，参考R波逻辑）
                        q_roi_y = np.argmax(q_green_channel)
                        q_global_y = y_sl + q_roi_y  # Q点Y坐标（全局，使用与R波相同的偏移-9）
                    s_point = min(abs_x + 15, img_rgb.shape[1] - 1)  # S点X坐标（全局）
                    s_roi_x = s_point - x_sl  # S点X坐标（ROI内）
                    if 0 <= s_roi_x < w_sl:  # 确保S点在ROI范围内
                        s_green_channel = spectral_roi[:, s_roi_x, 1]
                        s_roi_y = np.argmax(s_green_channel)
                        s_global_y = y_sl + s_roi_y  # S点Y坐标（全局，使用与R波相同的偏移-9）
                    # 绘制Q点竖线（黄色虚线）
                    ax4.axvline(x=q_point, color='yellow', linestyle='--', linewidth=1.2, alpha=0.7,
                                label='Q点' if peak == peaks[0] else "")
                # 新增：绘制第二个下坡点
                for down_point in second_downslopes:
                    abs_x = down_point
                    ax4.axvline(x=abs_x, color='cyan', linestyle='--', linewidth=1.5, alpha=0.8,
                                label='第二下坡点' if down_point == second_downslopes[0] else "")
            if current_smoothed_x is not None and current_smoothed_y is not None:
                ax4.plot(
                    current_smoothed_x, current_smoothed_y,
                    color='cyan',  # 青色曲线区分二次平滑
                    linewidth=1.5,
                    path_effects=[pe.Stroke(linewidth=4, foreground='blue'), pe.Normal()],  # 蓝边效果增强区分度
                    alpha=0.8,
                    label='二次平滑峰值波形包络线'
                )
            if peak_speed is not None and mean_speed_in_peak is not None:
                plt.annotate(
                    f'Peak Velocity: {peak_speed:.1f} cm/s\nMean Velocity: {mean_speed_in_peak:.1f} cm/s\nVTI:{vti:.1f}cm',
                    xy=(peak_x, peak_global_y),
                    xytext=(peak_x + 50, peak_global_y - 50),
                    fontsize=12,
                    color='white',
                    bbox=dict(facecolor='red', alpha=0.8),
                    arrowprops=dict(facecolor='yellow', shrink=0.05)
                )
            ax4.axhline(y=peak_global_y, color='y', linestyle='--', linewidth=2, alpha=0.7)
            x_vals = np.linspace(x, x + w - 1, len(upper_envelope))
            valid_mask = ~np.isnan(upper_envelope) & ~np.isnan(lower_envelope)
            ax4.plot(x_vals[valid_mask], (y + upper_envelope)[valid_mask], 'r-', lw=2.5, alpha=0.9)  # label='上包络线'
            ax4.plot(x_vals[valid_mask], (y + lower_envelope)[valid_mask], 'b-', lw=2.5, alpha=0.9)  # label='下包络
            # title_str = "DSE-AutoSTV"  # "多普勒频谱分析与峰值定位"
            # plt.title(title_str, pad=20, fontsize=15)
            ax4.set_xticks([])  # 移除X轴刻度
            ax4.set_yticks([])  # 移除Y轴刻度
            ax4.spines['top'].set_visible(False)  # 隐藏上边框
            ax4.spines['bottom'].set_visible(False)  # 隐藏下边框
            ax4.spines['left'].set_visible(False)  # 隐藏左边框
            ax4.spines['right'].set_visible(False)  # 隐藏右边框
            plt.tight_layout()
        return fig,fig1,fig2,fig3,fig4
    pass
class BatchDopplerAnalyzer:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.results = []  # 存储格式: (时间, 峰值速度, 平均速度, VTI)

    def analyze_folder(self):
        """分析文件夹中的所有图像"""
        analyzer = DopplerEnvelopeAnalyzer(
            prf=17,
            spectrum_region_ratio=(0.49, 1.0),
            baseline_exclude_height=40,
            text_region_width_ratio=0.1,
            left_text_width_ratio=0.14,
            left_text_height_ratio=0.66,
            speed_scale_height_ratio=(0.43, 0.955),
            speed_scale_width_ratio=0.07
        )
        raw_results = []
        for filename in sorted(os.listdir(self.folder_path)):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(self.folder_path, filename)
                try:
                    analyzer.speed_values = []
                    analyzer.speed_positions = []
                    img_gray, img_rgb = analyzer.load_image(image_path, return_original=True)
                    if img_gray is None or img_rgb is None:
                        continue
                    speed_scale_roi, scale_left_col, scale_top, _, _ = analyzer.extract_speed_scale_region(img_rgb)

                    # 检查是否成功识别速度刻度
                    if not analyzer.speed_values or len(analyzer.speed_values) < 2:
                        # print(f"警告: 无法识别速度刻度 - {filename}")
                        continue

                    # 提取时间信息
                    time_str = analyzer.extract_time_info(img_rgb)

                    # 定位频谱区域
                    x, y, w, h_roi = analyzer.locate_spectrum(img_gray)
                    spectrum_roi = img_gray[y:y + h_roi, x:x + w]
                    upper_envelope, lower_envelope = analyzer.extract_envelopes(spectrum_roi,smooth_method='moving_avg')

                    # 找到峰值位置
                    valid_upper = upper_envelope[~np.isnan(upper_envelope)]
                    if len(valid_upper) == 0:
                        continue

                    peak_y_idx = np.nanargmin(upper_envelope)
                    peak_y_roi = upper_envelope[peak_y_idx]
                    peak_x = x + peak_y_idx  # 峰值在原图x坐标
                    peak_global_y = y + peak_y_roi  # 峰值在原图y坐标

                    # 提取速度刻度
                    _, scale_left_col, scale_top, _, _ = analyzer.extract_speed_scale_region(img_rgb)
                    # 计算峰值速度
                    peak_y_relative = peak_global_y - scale_top
                    peak_speed = analyzer.get_speed_from_relative_position(peak_y_relative)

                    spectral_region = analyzer.detect_spectral_line(img_rgb)
                    peaks, smoothed, second_downslopes = analyzer.analyze_spectral_peaks(img_rgb, spectral_region)

                    peaks_left_shifted = np.clip(peaks - 15, 0, img_rgb.shape[1] - 1)
                    peak_envelope_x, peak_envelope_y, right_endpoints = analyzer.trace_peak_envelope(
                        upper_envelope, spectrum_roi, peak_y_idx, img_rgb, x, y, peaks_left_shifted, second_downslopes
                    )
                    part_endpoints = analyzer.detect_concave_endpoints(
                        peak_envelope_x, peak_envelope_y, min_segment_length=5
                    )
                    concave_endpoints = right_endpoints.copy()
                    for ep in part_endpoints:
                        if not any(abs(ep[0] - fe[0]) < 2 for fe in right_endpoints):  # X坐标差<2视为同一端点
                            concave_endpoints.append(ep)

                    smoothed_x, smoothed_y = analyzer.smooth_connect_anchors(
                        peak_envelope_x, peak_envelope_y, concave_endpoints
                    )
                    peak_point = (peak_x, peak_global_y, 'right')
                    current_smoothed_x, current_smoothed_y = smoothed_x.copy(), smoothed_y.copy()

                    for iter in range(1, 7):
                        # 1. 检测当前曲线的凹区间端点
                        concave_pts = analyzer.detect_concave_endpoints(
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
                        current_smoothed_x, current_smoothed_y = analyzer.smooth_connect_anchors(
                            current_smoothed_x,
                            current_smoothed_y,
                            concave_pts
                        )
                    mean_speed = analyzer.calculate_average_velocity(current_smoothed_x, current_smoothed_y, img_rgb,
                                                                         scale_top)

                    vti, time_interval = analyzer.calculate_velocity_time_integral(
                        current_smoothed_x, current_smoothed_y, img_rgb, scale_top)

                    if all(v is not None for v in [peak_speed, mean_speed, vti]):
                        # 时间格式处理
                        try:
                            time_obj = datetime.strptime(time_str, "%H:%M:%S")
                        except ValueError:
                            try:
                                time_obj = datetime.strptime(time_str, "%H:%M")
                            except:
                                time_obj = datetime.strptime("00:00:00", "%H:%M:%S")  # 默认时间

                        raw_results.append((filename, time_obj, peak_speed, mean_speed, vti))
                        print(f"处理完成: {filename} | 时间: {time_obj.time()} | "
                              f"峰值: {peak_speed:.1f} cm/s | "
                              f"均值: {mean_speed:.1f} cm/s | "
                              f"VTI: {vti:.1f} cm")

                except Exception as e:
                    print(f"处理 {filename} 时出错: {str(e)}")
                    continue
            # 提取结果并按时间排序
        self.results = [r[1:] for r in raw_results]  # 去掉文件名
        self.results.sort(key=lambda x: x[0])
        # 按时间排序并截取前20分钟数据
        if self.results:
            self.results.sort(key=lambda x: x[0])

        return self.results


    def plot_combined_curves(self, output_path=None, duration_minutes=20):
        """分别在三张独立画布上绘制峰值速度、平均速度、VTI的折线图"""
        if not self.results or len(self.results) < 2:
            print("有效数据不足，至少需要2个数据点")
            return

        # 准备数据
        base_time = self.results[0][0]
        times_min = [(r[0] - base_time).total_seconds() / 60 for r in self.results]
        peak_speeds = [r[1] for r in self.results]
        mean_speeds = [r[2] for r in self.results]
        vtis = [r[3] for r in self.results]

        def smooth_data(data, times):
            smoothed = data.copy()
            for i in range(1, len(data) - 1):
                if data[i] < data[i - 1] and data[i] < data[i + 1]:
                    smoothed[i] = (data[i - 1] + data[i + 1]) / 2
            return smoothed

        smoothed_peak = smooth_data(peak_speeds, times_min)
        smoothed_peak = smooth_data(smoothed_peak, times_min)
        smoothed_peak = smooth_data(smoothed_peak, times_min)
        smoothed_peak = smooth_data(smoothed_peak, times_min)
        smoothed_peak = smooth_data(smoothed_peak, times_min)
        smoothed_peak = smooth_data(smoothed_peak, times_min)

        # 平均速度 - 保持原有平滑逻辑（仅处理极小值）
        smoothed_mean = smooth_data(mean_speeds, times_min)
        smoothed_mean = smooth_data(smoothed_mean, times_min)
        smoothed_mean = smooth_data(smoothed_mean, times_min)
        smoothed_mean = smooth_data(smoothed_mean, times_min)
        smoothed_mean = smooth_data(smoothed_mean, times_min)

        # VTI - 使用新的平滑逻辑（同时处理极大值和极小值）
        smoothed_vti = smooth_data(vtis, times_min)
        smoothed_vti = smooth_data(smoothed_vti, times_min)
        smoothed_vti = smooth_data(smoothed_vti, times_min)
        smoothed_vti = smooth_data(smoothed_vti, times_min)
        smoothed_vti = smooth_data(smoothed_vti, times_min)
        smoothed_vti = smooth_data(smoothed_vti, times_min)

        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(times_min, smoothed_peak, 'o-', color='tab:red', linewidth=1.5, markersize=6)
        ax1.set_xlabel("Time (min)")
        ax1.set_ylabel("Peak Velocity (cm/s)")
        ax1.set_title("Peak Velocity Temporal Variation")
        ax1.grid(True, linestyle='--', alpha=0.6)

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(times_min, smoothed_mean, 'o-', color='tab:blue', linewidth=1.5, markersize=6)
        ax2.set_xlabel("Time (min)")
        ax2.set_ylabel("Mean Velocity (cm/s)")
        ax2.set_title("Mean Velocity Temporal Variation")
        ax2.grid(True, linestyle='--', alpha=0.6)

        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ax3.plot(times_min, smoothed_vti, 'o-', color='tab:green', linewidth=1.5, markersize=6)
        ax3.set_xlabel("Time (min)")
        ax3.set_ylabel("VTI (cm)")
        ax3.set_title("VTI Temporal Variation")
        ax3.grid(True, linestyle='--', alpha=0.6)

        fig4,ax4=plt.subplots(figsize=(10, 5))
        ax4.plot(times_min, smoothed_peak, 'o-', color='tab:red', linewidth=1.5, markersize=6,
                 label='Peak Velocity (cm/s)')  # 直接在 label 中包含单位
        ax4.plot(times_min, smoothed_mean, 'o-', color='tab:blue', linewidth=1.5, markersize=6,
                 label='Mean Velocity (cm/s)')
        ax4.plot(times_min, smoothed_vti, 'o-', color='tab:green', linewidth=1.5, markersize=6,
                 label='VTI (cm)')
        ax4.set_xlabel("Time (min)")  # 横坐标标签
        ax4.set_ylabel("Velocity (cm/s) / VTI (cm)")  # 纵坐标标签
        ax4.legend(fontsize=12, loc='upper right')
        ax4.set_title("Multiple Indicators Temporal Variation")
        ax4.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        return fig1, fig2, fig3, fig4


def main():
    st.set_page_config(page_title="DSE-AutoSTV", layout="wide")

    st.markdown("""
           <style>
           /* 侧边栏背景色 */
           [data-testid="stSidebar"] {
               background-color: #AAD4EC !important;
           }

           /* 侧边栏标题样式 */
           [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 {
               color: #2C3E50;
               font-family: 'Segoe UI', sans-serif;
           }

           /* 下拉选择框样式 */
           .stSelectbox [data-baseweb="select"] {
               background-color: white;
               border-radius: 8px;
               border: 1px solid #CBD5E1;
               box-shadow: 0 2px 5px rgba(0,0,0,0.05);
           }

           /* 下拉选项悬停样式 */
           .stSelectbox [data-baseweb="select"] [role="listbox"] {
               background-color: white;
               border-radius: 8px;
               box-shadow: 0 4px 12px rgba(0,0,0,0.1);
               border: none;
           }

           .stSelectbox [data-baseweb="select"] [role="option"]:hover {
               background-color: #E6F7FF;
               color: #1890FF;
           }

           /* 按钮样式 */
           .stButton button {
               background-color: #1890FF;
               color: white;
               border-radius: 8px;
               border: none;
               padding: 0.6rem 1.2rem;
               font-weight: 500;
               transition: all 0.3s ease;
           }

           .stButton button:hover {
               background-color: #096DD9;
               box-shadow: 0 4px 8px rgba(24, 144, 255, 0.3);
           }

           /* 页面标题样式 */
           .stTitle {
               color: #2C3E50;
               font-family: 'Segoe UI', sans-serif;
               padding-bottom: 0.5rem;
               border-bottom: 2px solid #AAD4EC;
           }

           /* 步骤卡片样式 */
           .step-card {
               background: white;
               border-radius: 12px;
               padding: 20px;
               box-shadow: 0 4px 12px rgba(0,0,0,0.08);
               margin-bottom: 25px;
               border-left: 5px solid #AAD4EC;
           }
           </style>
           """, unsafe_allow_html=True)

    st.title("Fully Automated Spectral-Temporal-Velocity Estimation Framework of Doppler Stress Echocardiography")
    analyzer = DopplerEnvelopeAnalyzer(
        prf=17,
        spectrum_region_ratio=(0.49, 1.0),
        baseline_exclude_height=40,
        text_region_width_ratio=0.1,
        left_text_width_ratio=0.14,
        left_text_height_ratio=0.65,
        speed_scale_height_ratio=(0.43, 0.955),
        speed_scale_width_ratio=0.07
    )

    # 初始化session state变量
    if 'step' not in st.session_state:
        st.session_state.step = 0
    if 'process_started' not in st.session_state:
        st.session_state.process_started = False
    if 'step1_complete' not in st.session_state:
        st.session_state.step1_complete = False
    if 'step2_complete' not in st.session_state:
        st.session_state.step2_complete = False
    if 'step3_complete' not in st.session_state:
        st.session_state.step3_complete = False
    if 'step4_complete' not in st.session_state:
        st.session_state.step4_complete = False
    if 'figures' not in st.session_state:
        st.session_state.figures = {}

    # 左侧参数选择栏
    with st.sidebar:
        st.markdown("""
            <style>
            /* 堆叠图像容器：移除固定高度，使用相对定位 */
            .image-stack {
                position: relative;
                margin-bottom: 5px;
            }
            /* 堆叠图像样式：调整偏移量，避免溢出 */
            .stacked-img {
                position: absolute;
                border: 1px solid #ddd;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
            }
            /* 标题样式：确保类名正确对应 */
            h1.custom-heading {
                font-size: 24px;
                font-weight: 700;
                color: #2C3E50;
                margin-bottom: 20px;
                border-bottom: 2px solid #AAD4EC;
                padding-bottom: 8px;
            }
            /* 处理步骤容器：确保在堆叠图像下方 */
            .process-steps {
                margin-top: 300px;
                position: relative;
                z-index: 10;
            }
            </style>

            <h1 class='custom-heading'>📊 Image & Process Flow</h1>
            """, unsafe_allow_html=True)

        # 1. 原始图像展示
        _path = r"D:\雨婷\超声多普勒分析\jpg_results\3.png"
        folder_path = r"D:\雨婷\超声多普勒分析\jpg图片\精简\0409精简"

        image_paths = [
            r"D:\雨婷\超声多普勒分析\jpg图片\原图\0409\IM_0358.jpg",
            r"D:\雨婷\超声多普勒分析\jpg图片\原图\0409\IM_0359.jpg",
            r"D:\雨婷\超声多普勒分析\jpg图片\原图\0409\IM_0361.jpg",
            r"D:\雨婷\超声多普勒分析\jpg图片\原图\0409\IM_0362.jpg"
        ]

        # 检查所有图片是否存在
        missing_paths = [p for p in image_paths if not os.path.exists(p)]
        if missing_paths:
            st.error(f"❌ 缺失图片: {', '.join(missing_paths)}")
        else:
            # 读取所有图片并转换为base64
            base64_images = []
            for path in image_paths:
                with open(path, "rb") as f:
                    base64_images.append(base64.b64encode(f.read()).decode("utf-8"))

            # 显示标题
            st.markdown("### Original Doppler Images")

            # Create stacking container
            col = st.columns(1)[0]
            with col:
                st.markdown('<div class="image-stack">', unsafe_allow_html=True)

                # Display images in reverse order (first image at bottom)
                for i, img_b64 in enumerate(reversed(base64_images)):
                    left = i * 20
                    top = i * 15
                    z_index = i

                    st.markdown(f"""
                            <img src="data:image/jpg;base64,{img_b64}" 
                                 class="stacked-img"
                                 style="left: {left}px; top: {top}px; z-index: {z_index}; 
                                        width: 90%; max-width: 600px; position: absolute;">
                        """, unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

        if os.path.exists(_path):
            with open(_path, "rb") as img_file:
                img_bytes = img_file.read()
                img_base64 = base64.b64encode(img_bytes).decode("utf-8")

                st.markdown("""
                                <div class="process-steps">
                                    <div class="image-card" style="padding-top: 5px; margin-bottom: 100px;">
                                        <h3 style="color: #2C3E50; margin-top: 0; font-family: 'Segoe UI', sans-serif;">Processing Steps</h3>
                                        <img src="data:image/jpg;base64,{}" style="width:100%; border-radius:5px; margin-top:10px;">
                                    </div>
                                </div>
                            """.format(img_base64), unsafe_allow_html=True)
        else:
            st.error(f"Processing steps image not found: {_path}")

        # 主处理按钮
        if st.button("🚀 Start Processing"):
            st.session_state.process_started = True
            st.session_state.step = 1
            st.rerun()

    # 主内容区域
    if st.session_state.process_started:
        # 步骤1卡片
        with st.container():
            st.markdown("""
                <div class="step-card">
                    <h3 style="color: #2C3E50; margin-top: 0; font-family: 'Segoe UI', sans-serif;">
                    Step 1: Scale-Aware Heatmap Network for Multimodal Cardiac Cycle Segmentation
                    </h3>
                    <p style="color: #666; margin-bottom: 15px;">
                    This step shows the heatmap visualization and TR-peak detection results of the Doppler image.
                    </p>
            """, unsafe_allow_html=True)

            def img_to_b64(path: str) -> str:
                with open(path, "rb") as f:
                    return base64.b64encode(f.read()).decode()

            def fig_to_b64(fig):
                buf = BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
                buf.seek(0)
                return base64.b64encode(buf.read()).decode()
            if st.session_state.step >= 1:
                with st.container():
                    if "fig1" not in st.session_state.figures:
                        with st.spinner("Processing Step 1..."):
                            fig, fig1, fig2, fig3, fig4 = analyzer.visualize(
                                r"D:\雨婷\超声多普勒分析\heatmap\val2017\50560820240722_LY-FUHE-KONGGE_20240722090753241.jpg"
                            )
                            st.session_state.figures.update({"fig": fig, "fig1": fig1, "fig2": fig2,
                                                             "fig3": fig3, "fig4": fig4})
                            st.rerun()
                    else:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"""
                                    <div style="background:white; border-radius:8px; padding:15px;">
                                        <h4>📈 Heatmap</h4>
                                        <img src="data:image/png;base64,{img_to_b64(r"D:/雨婷/超声多普勒分析/heatmap/D/one/6.jpg")}" style="width:100%; border-radius:5px;">
                                    </div>
                                """, unsafe_allow_html=True)
                        with col2:
                            st.markdown(f"""
                                    <div style="background:white; border-radius:8px; padding:15px;">
                                        <h4>🔍 Detection Results</h4>
                                        <img src="data:image/png;base64,{img_to_b64(r'D:/雨婷/超声多普勒分析/heatmap/results新/result_valid_6.jpg')}" style="width:100%; border-radius:5px;">
                                    </div>
                                """, unsafe_allow_html=True)
                        st.success("Step 1 completed!")
                        if st.session_state.step == 1:
                            st.session_state.step = 2
                            st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)

                # ---------- 5. 步骤2 ----------
                # ---------- 5. 步骤2（完全还原你原来的小卡片 + 子步骤标题） ----------
                if st.session_state.step >= 2:
                    with st.container():
                        st.markdown("""
                            <div class="step-card">
                                <h3 style="color: #2C3E50; margin-top: 0; font-family: 'Segoe UI', sans-serif;">
                                Step 2: Physiological Timed Envelope Tracking
                                </h3>
                                <p style="color: #666; margin-bottom: 15px;">
                                This step shows the cluster selection and envelope processing results step by step.
                                </p>
                        """, unsafe_allow_html=True)

                        IMAGE_CONTAINER = """
                            <div style="background: white; border-radius: 8px; padding: 15px; margin-bottom: 20px; 
                                        box-shadow: 0 2px 4px rgba(0,0,0,0.1); max-width: 800px; margin-left: auto; margin-right: auto;">
                                <h4 style="color: #2C3E50; margin-top: 0; margin-bottom: 15px;">{subplot_title}</h4>
                                <img src="data:image/png;base64,{image_data}" style="width:100%; border-radius:5px;">
                            </div>
                        """
                        SUBSTEP_HEADER = """
                            <div style="display: flex; align-items: center; margin-bottom: 15px; margin-top: 30px;">
                                <div style="background-color: #f0f7ff; color: #1890FF; padding: 6px 12px; border-radius: 20px; 
                                            font-weight: 600; margin-right: 12px; font-size: 22px;">
                                    Substep {num}
                                </div>
                                <h4 style="color: #2C3E50; margin: 0; font-size: 22px;">{title}</h4>
                            </div>
                        """

                        # ---------- 2.1 ----------
                        st.markdown(SUBSTEP_HEADER.format(num="1",
                                                          title="Spatial Density-Based Clustering + Cluster selection"),
                                    unsafe_allow_html=True)
                        if "cluster" not in st.session_state.figures:
                            with st.spinner("Processing Substep 2.1..."):
                                st.session_state.figures["cluster"] = img_to_b64(
                                    r"D:\雨婷\超声多普勒分析\jpg_results\聚类选择.png")
                                st.rerun()
                        st.markdown(IMAGE_CONTAINER.format(
                            subplot_title="📍 Cluster Selection Result",
                            image_data=st.session_state.figures["cluster"]
                        ), unsafe_allow_html=True)

                        # ---------- 2.2 ----------
                        st.markdown(SUBSTEP_HEADER.format(num="2",
                                                          title="Envelope Extraction + 1D Conv Smoothing + Waveform Completion with Bezier Curves"),
                                    unsafe_allow_html=True)
                        if "envelope" not in st.session_state.figures:
                            with st.spinner("Processing Substep 2.2..."):
                                st.session_state.figures["envelope"] = fig_to_b64(st.session_state.figures["fig3"])
                                st.rerun()
                        st.markdown(IMAGE_CONTAINER.format(
                            subplot_title="🧩 Envelope Completion Result",
                            image_data=st.session_state.figures["envelope"]
                        ), unsafe_allow_html=True)

                        # ---------- 2.3 ----------
                        st.markdown(SUBSTEP_HEADER.format(num="3", title="Parameter Extraction"),
                                    unsafe_allow_html=True)
                        if "parameters" not in st.session_state.figures:
                            with st.spinner("Processing Substep 2.3..."):
                                st.session_state.figures["parameters"] = fig_to_b64(st.session_state.figures["fig4"])
                                st.rerun()
                        st.markdown(IMAGE_CONTAINER.format(
                            subplot_title="🔬 Parameter Extraction Result",
                            image_data=st.session_state.figures["parameters"]
                        ), unsafe_allow_html=True)

                        st.success("Step 2 completed successfully!")
                        if st.session_state.step == 2:
                            st.session_state.step = 3
                            st.rerun()
                        st.markdown("</div>", unsafe_allow_html=True)

                # ---------- 6. 步骤3 ----------
            if st.session_state.step >= 3:
                with st.container():
                    st.markdown("""
                            <div class="step-card">
                                <h3>Step 3: Dynamic Angle Correction</h3>
                                <p style="color:#666;">Sampling line extraction, blood region detection and angle calculation.</p>
                        """, unsafe_allow_html=True)
                    if "angle" not in st.session_state.figures:
                        with st.spinner("Processing Step 3..."):
                            st.session_state.figures["angle"] = img_to_b64(
                                r"D:\雨婷\超声多普勒分析\jpg_results\角度.png")
                            st.rerun()
                    st.markdown(f"""
                            <div style="background:white; border-radius:8px; padding:20px; max-width:800px; margin:auto;">
                                <h4>⚙️ Dynamic Angle Correction Result</h4>
                                <img src="data:image/png;base64,{st.session_state.figures['angle']}" style="width:100%; border-radius:5px;">
                            </div>
                        """, unsafe_allow_html=True)
                    st.success("Step 3 completed!")
                    if st.session_state.step == 3:
                        st.session_state.step = 4
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)

                # ---------- 7. 步骤4 ----------
            # ---------- 7. 步骤4（完全还原你原来的 2×2 小卡片布局） ----------
            if st.session_state.step >= 4:
                with st.container():
                    st.markdown("""
                        <div class="step-card">
                            <h3 style="color: #2C3E50; margin-top: 0; font-family: 'Segoe UI', sans-serif;">
                            Step 4: Parameter Time-Series Curves
                            </h3>
                            <p style="color: #666; margin-bottom: 15px;">
                            This step shows the final parameter time-series curves analysis results.
                            </p>
                    """, unsafe_allow_html=True)

                    if "combined_fig1" not in st.session_state.figures:
                        with st.spinner("Processing Step 4..."):
                            batch_analyzer = BatchDopplerAnalyzer(folder_path)
                            results = batch_analyzer.analyze_folder()
                            if results:
                                f1, f2, f3, f4 = batch_analyzer.plot_combined_curves()
                                st.session_state.figures.update({
                                    "combined_fig1": f1, "combined_fig2": f2,
                                    "combined_fig3": f3, "combined_fig4": f4
                                })
                                st.session_state.results = results
                                st.rerun()
                            else:
                                st.error("Step 4 failed – no results.")

                    # --- 2×2 小卡片 ---
                    COL_STYLE = """
                        <div style="background: white; border-radius: 8px; padding: 15px; margin-bottom: 20px;
                                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                            <h4 style="color: #2C3E50; margin-top: 0; margin-bottom: 10px;">{title}</h4>
                            <img src="data:image/png;base64,{img}" style="width:100%; border-radius:5px;">
                        </div>
                    """
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(COL_STYLE.format(
                            title="Peak Velocity Temporal Variation",
                            img=fig_to_b64(st.session_state.figures["combined_fig1"])
                        ), unsafe_allow_html=True)
                        st.markdown(COL_STYLE.format(
                            title="Mean Velocity Temporal Variation",
                            img=fig_to_b64(st.session_state.figures["combined_fig2"])
                        ), unsafe_allow_html=True)
                    with c2:
                        st.markdown(COL_STYLE.format(
                            title="Velocity Time Integral Variation",
                            img=fig_to_b64(st.session_state.figures["combined_fig3"])
                        ), unsafe_allow_html=True)
                        st.markdown(COL_STYLE.format(
                            title="Heart Rate Temporal Variation",
                            img=fig_to_b64(st.session_state.figures["combined_fig4"])
                        ), unsafe_allow_html=True)

                    st.success("All steps completed!")
                    st.markdown("</div>", unsafe_allow_html=True)

        # ---------- 工具 ----------


        #     if not st.session_state.step1_complete:
        #         if st.button("Run Step 1", key="step1_button"):
        #             with st.spinner("Processing Step 1..."):
        #                 image_path = r"D:\雨婷\超声多普勒分析\heatmap\val2017\50560820240722_LY-FUHE-KONGGE_20240722090753241.jpg"
        #                 fig, fig1, fig2, fig3, fig4 = analyzer.visualize(image_path)
        #
        #                 st.session_state.figures['fig'], st.session_state.figures['fig1'], \
        #                     st.session_state.figures['fig2'], st.session_state.figures['fig3'], \
        #                     st.session_state.figures['fig4'] = analyzer.visualize(image_path)
        #
        #                 if st.session_state.figures['fig'] and st.session_state.figures['fig1']:
        #                     st.session_state.step1_complete = True
        #                     st.session_state.step = 2
        #                     st.rerun()
        #                 else:
        #                     st.error("Step 1 processing failed")
        #
        #     if st.session_state.step1_complete:
        #         def local_img_to_base64(path):
        #             import base64
        #             with open(path, "rb") as f:
        #                 return base64.b64encode(f.read()).decode()
        #
        #         fig1_base64 = local_img_to_base64(r"D:\雨婷\超声多普勒分析\heatmap\D\one\6.jpg")
        #         fig2_base64 = local_img_to_base64(r"D:\雨婷\超声多普勒分析\heatmap\results新\result_valid_6.jpg")
        #
        #         col1, col2 = st.columns(2)
        #         with col1:
        #             st.markdown(f"""
        #                 <div style="background: white; border-radius: 8px; padding: 15px; margin-bottom: 15px;">
        #                     <h4 style="color: #2C3E50; margin-top: 0;">📈 Heatmap</h4>
        #                     <img src="data:image/png;base64,{fig1_base64}" style="width:100%; border-radius:5px;">
        #                 </div>
        #             """, unsafe_allow_html=True)
        #
        #         with col2:
        #             st.markdown(f"""
        #                 <div style="background: white; border-radius: 8px; padding: 15px; margin-bottom: 15px;">
        #                     <h4 style="color: #2C3E50; margin-top: 0;">🔍 Detection Results</h4>
        #                     <img src="data:image/png;base64,{fig2_base64}" style="width:100%; border-radius:5px;">
        #                 </div>
        #             """, unsafe_allow_html=True)
        #
        #         st.success("Step 1 completed successfully!")
        #         st.markdown("</div>", unsafe_allow_html=True)
        #     else:
        #         st.markdown("</div>", unsafe_allow_html=True)
        #
        # # 步骤2卡片 (只有在步骤1完成后显示)
        # if st.session_state.step >= 2:
        #     with st.container():
        #         st.markdown("""
        #             <div class="step-card" style="padding: 20px;">
        #                 <h3 style="color: #2C3E50; margin-top: 0; font-family: 'Segoe UI', sans-serif;">
        #                 Step 2: Physiological Timed Envelope Tracking
        #                 </h3>
        #                 <p style="color: #666; margin-bottom: 15px;">
        #                 This step shows the cluster selection and envelope processing results step by step.
        #                 </p>
        #         """, unsafe_allow_html=True)
        #
        #         # 初始化步骤2内部状态
        #         if 'step2_substep' not in st.session_state:
        #             st.session_state.step2_substep = 1
        #         if 'step2_figures' not in st.session_state:
        #             st.session_state.step2_figures = {}
        #
        #         # 统一的图像容器样式 (参考步骤3的卡片大小)
        #         IMAGE_CONTAINER = """
        #             <div style="background: white; border-radius: 8px; padding: 15px; margin-bottom: 20px;
        #                         box-shadow: 0 2px 4px rgba(0,0,0,0.1); max-width: 800px; margin-left: auto; margin-right: auto;">
        #                 <h4 style="color: #2C3E50; margin-top: 0; margin-bottom: 15px;">{subplot_title}</h4>
        #                 <img src="data:image/png;base64,{image_data}" style="width:100%; border-radius:5px;">
        #             </div>
        #         """
        #
        #         SUBSTEP_HEADER = """
        #                     <div style="display: flex; align-items: center; margin-bottom: 15px; margin-top: 30px;">
        #                         <div style="background-color: #f0f7ff; color: #1890FF; padding: 6px 12px; border-radius: 20px;
        #                                     font-weight: 600; margin-right: 12px; font-size: 22px;">
        #                             Substep {num}
        #                         </div>
        #                         <h4 style="color: #2C3E50; margin: 0; font-size: 22px;">{title}</h4>
        #                     </div>
        #                 """
        #         # 子步骤1: 聚类选择
        #         # st.markdown("""
        #         #     <div style="margin-bottom: 30px;">
        #         #         <div style="display: flex; align-items: center; margin-bottom: 10px;">
        #         #             <div style="background-color: #1890FF; color: white; width: 24px; height: 24px; border-radius: 50%;
        #         #                         display: flex; justify-content: center; align-items: center; margin-right: 10px;">1</div>
        #         #             <h4 style="color: #2C3E50; margin: 0;">Cluster Selection</h4>
        #         #         </div>
        #         # """, unsafe_allow_html=True)
        #         st.markdown(SUBSTEP_HEADER.format(
        #             num="1",
        #             title="Spatial Density-Based Clustering+Cluster selection"
        #         ), unsafe_allow_html=True)
        #
        #         if 'cluster' not in st.session_state.step2_figures:
        #             if st.button("Run Substep1", key="step2_sub1_button"):
        #                 with st.spinner("Processing Substep 1..."):
        #                     st.session_state.step2_figures['cluster'] = local_img_to_base64(
        #                         r"D:\雨婷\超声多普勒分析\jpg_results\聚类选择.png"
        #                     )
        #                     st.session_state.step2_substep = 2
        #                     st.rerun()
        #         else:
        #             st.markdown(IMAGE_CONTAINER.format(
        #                 subplot_title="📍 Cluster Selection Result",
        #                 image_data=st.session_state.step2_figures['cluster']
        #             ), unsafe_allow_html=True)
        #
        #         st.markdown("</div>", unsafe_allow_html=True)
        #
        #         # 子步骤2: 包络完成 (只在子步骤1完成后显示)
        #         if st.session_state.step2_substep >= 2:
        #             st.markdown(SUBSTEP_HEADER.format(
        #                 num="2",
        #                 title="Envelope Extraction+ 1D Cony Smoothing+WaveformCompletion withBezier Curyes"
        #             ), unsafe_allow_html=True)
        #
        #             if 'envelope' not in st.session_state.step2_figures:
        #                 if st.button("Run Substep2", key="step2_sub2_button"):
        #                     with st.spinner("Processing Substep 2..."):
        #                         def fig_to_base64(fig):
        #                             buf = BytesIO()
        #                             fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        #                             buf.seek(0)
        #                             return base64.b64encode(buf.getvalue()).decode("utf-8")
        #
        #                         st.session_state.step2_figures['envelope'] = fig_to_base64(
        #                             st.session_state.figures['fig3']
        #                         )
        #                         st.session_state.step2_substep = 3
        #                         st.rerun()
        #             else:
        #                 st.markdown(IMAGE_CONTAINER.format(
        #                     subplot_title="🧩 Envelope Completion Result",
        #                     image_data=st.session_state.step2_figures['envelope']
        #                 ), unsafe_allow_html=True)
        #
        #             st.markdown("</div>", unsafe_allow_html=True)
        #
        #         # 子步骤3: 参数提取 (只在子步骤2完成后显示)
        #         if st.session_state.step2_substep >= 3:
        #             st.markdown(SUBSTEP_HEADER.format(
        #                 num="3",
        #                 title="Parameter Extraction"
        #             ), unsafe_allow_html=True)
        #
        #             if 'parameters' not in st.session_state.step2_figures:
        #                 if st.button("Run Substep3", key="step2_sub3_button"):
        #                     with st.spinner("Processing Substep 3..."):
        #                         def fig_to_base64(fig):
        #                             buf = BytesIO()
        #                             fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        #                             buf.seek(0)
        #                             return base64.b64encode(buf.getvalue()).decode("utf-8")
        #
        #                         st.session_state.step2_figures['parameters'] = fig_to_base64(
        #                             st.session_state.figures['fig4']
        #                         )
        #                         st.session_state.step2_complete = True
        #                         st.session_state.step = 3
        #                         st.rerun()
        #             else:
        #                 st.markdown(IMAGE_CONTAINER.format(
        #                     subplot_title="🔬 Parameter Extraction Result",
        #                     image_data=st.session_state.step2_figures['parameters']
        #                 ), unsafe_allow_html=True)
        #                 st.success("Step 2 completed successfully!")
        #
        #             st.markdown("</div>", unsafe_allow_html=True)
        #
        #         st.markdown("</div>", unsafe_allow_html=True)
        # # 步骤3卡片 (只有在步骤2完成后显示)
        # if st.session_state.step >= 3:
        #     with st.container():
        #         st.markdown("""
        #                             <div class="step-card" style="padding: 20px;">
        #                                 <h3 style="color: #2C3E50; margin-top: 0; font-family: 'Segoe UI', sans-serif;">
        #                                 Step 3: Angle Correction
        #                                 </h3>
        #                                 <p style="color: #666; margin-bottom: 15px;">
        #                                 This step shows the sampling line extraction,blood region detection and angle calculation.
        #                                 </p>
        #                         """, unsafe_allow_html=True)
        #
        #         if not st.session_state.step3_complete:
        #             if st.button("Run Step 3", key="step3_button"):
        #                 with st.spinner("Processing Step 3..."):
        #                     st.session_state.step3_complete = True
        #                     st.session_state.step = 4
        #                     st.rerun()
        #
        #         if st.session_state.step3_complete:
        #             fig6_base64 = local_img_to_base64(r"D:\雨婷\超声多普勒分析\jpg_results\角度.png")
        #
        #             st.markdown(f"""
        #                 <div style="background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px;
        #                             box-shadow: 0 2px 4px rgba(0,0,0,0.1); max-width: 800px; margin-left: auto; margin-right: auto;">
        #                     <h4 style="color: #2C3E50; margin-top: 0; margin-bottom: 15px; font-size: 18px;">⚙️ Real-time Angle Correction</h4>
        #                     <img src="data:image/png;base64,{fig6_base64}" style="width:100%; border-radius:5px;">
        #                 </div>
        #             """, unsafe_allow_html=True)
        #
        #             st.success("Step 3 completed successfully!")
        #             st.markdown("</div>", unsafe_allow_html=True)
        #         else:
        #             st.markdown("</div>", unsafe_allow_html=True)
        #
        # # 步骤4卡片 (只有在步骤3完成后显示)
        # if st.session_state.step >= 4:
        #     with st.container():
        #         st.markdown("""
        #               <div class="step-card">
        #                   <h3 style="color: #2C3E50; margin-top: 0; font-family: 'Segoe UI', sans-serif;">
        #                   Step 4: Parameter Time-Series Cuves
        #                   </h3>
        #                   <p style="color: #666; margin-bottom: 15px;">
        #                   This step shows the final parameter time-series cuves analysis results.
        #                   </p>
        #           """, unsafe_allow_html=True)
        #
        #         if not st.session_state.step4_complete:
        #             if st.button("Run Step 4", key="step4_button"):
        #                 with st.spinner("Processing Step 4..."):
        #                     # 确保使用BatchDopplerAnalyzer并正确调用方法
        #                     batch_analyzer = BatchDopplerAnalyzer(folder_path)
        #                     results = batch_analyzer.analyze_folder()
        #
        #                     if results:
        #                         try:
        #                             # 调用plot_combined_curves方法并获取返回的图表
        #                             fig1, fig2, fig3, fig4 = batch_analyzer.plot_combined_curves()
        #
        #                             # 将图表存储在session_state中
        #                             st.session_state.figures['combined_fig1'] = fig1
        #                             st.session_state.figures['combined_fig2'] = fig2
        #                             st.session_state.figures['combined_fig3'] = fig3
        #                             st.session_state.figures['combined_fig4'] = fig4
        #
        #                             st.session_state.step4_complete = True
        #                             st.rerun()
        #                         except Exception as e:
        #                             st.error(f"Error generating combined curves: {str(e)}")
        #                             # 输出详细错误信息以便调试
        #                             st.exception(e)
        #                     else:
        #                         st.error("Step 4 processing failed - no results generated")
        #
        #         if st.session_state.step4_complete and 'combined_fig1' in st.session_state.figures:
        #             def fig_to_base64(fig):
        #                 buf = BytesIO()
        #                 fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        #                 buf.seek(0)
        #                 return base64.b64encode(buf.getvalue()).decode("utf-8")
        #
        #             # 从session_state获取图表
        #             fig1_base64 = fig_to_base64(st.session_state.figures['combined_fig1'])
        #             fig2_base64 = fig_to_base64(st.session_state.figures['combined_fig2'])
        #             fig3_base64 = fig_to_base64(st.session_state.figures['combined_fig3'])
        #             fig4_base64 = fig_to_base64(st.session_state.figures['combined_fig4'])
        #
        #             st.markdown("""
        #                   <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px;">
        #                       <div style="width: 100%; height: 350px; display: flex; justify-content: center; align-items: center;">
        #                           <img src="data:image/png;base64,{fig1_base64}" style="max-width:100%; max-height:330px; border-radius:5px;">
        #                       </div>
        #                       <div style="width: 100%; height: 350px; display: flex; justify-content: center; align-items: center;">
        #                           <img src="data:image/png;base64,{fig2_base64}" style="max-width:100%; max-height:330px; border-radius:5px;">
        #                       </div>
        #                       <div style="width: 100%; height: 350px; display: flex; justify-content: center; align-items: center;">
        #                           <img src="data:image/png;base64,{fig3_base64}" style="max-width:100%; max-height:330px; border-radius:5px;">
        #                       </div>
        #                       <div style="width: 100%; height: 350px; display: flex; justify-content: center; align-items: center;">
        #                           <img src="data:image/png;base64,{fig4_base64}" style="max-width:100%; max-height:330px; border-radius:5px;">
        #                       </div>
        #                   </div>
        #               """.format(
        #                 fig1_base64=fig1_base64,
        #                 fig2_base64=fig2_base64,
        #                 fig3_base64=fig3_base64,
        #                 fig4_base64=fig4_base64
        #             ), unsafe_allow_html=True)
        #
        #             st.success("All steps completed successfully!")
        #             st.markdown("</div>", unsafe_allow_html=True)
        #         else:
        #             st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
# def main():
#     st.set_page_config(page_title="DSE-AutoSTV", layout="wide")
#
#     st.markdown("""
#            <style>
#            /* 侧边栏背景色 */
#            [data-testid="stSidebar"] {
#                background-color: #AAD4EC !important;
#            }
#
#            /* 侧边栏标题样式 */
#            [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 {
#                color: #2C3E50;
#                font-family: 'Segoe UI', sans-serif;
#            }
#
#            /* 下拉选择框样式 */
#            .stSelectbox [data-baseweb="select"] {
#                background-color: white;
#                border-radius: 8px;
#                border: 1px solid #CBD5E1;
#                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
#            }
#
#            /* 下拉选项悬停样式 */
#            .stSelectbox [data-baseweb="select"] [role="listbox"] {
#                background-color: white;
#                border-radius: 8px;
#                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
#                border: none;
#            }
#
#            .stSelectbox [data-baseweb="select"] [role="option"]:hover {
#                background-color: #E6F7FF;
#                color: #1890FF;
#            }
#
#            /* 按钮样式 */
#            .stButton button {
#                background-color: #1890FF;
#                color: white;
#                border-radius: 8px;
#                border: none;
#                padding: 0.6rem 1.2rem;
#                font-weight: 500;
#                transition: all 0.3s ease;
#            }
#
#            .stButton button:hover {
#                background-color: #096DD9;
#                box-shadow: 0 4px 8px rgba(24, 144, 255, 0.3);
#            }
#
#            /* 页面标题样式 */
#            .stTitle {
#                color: #2C3E50;
#                font-family: 'Segoe UI', sans-serif;
#                padding-bottom: 0.5rem;
#                border-bottom: 2px solid #AAD4EC;
#            }
#            </style>
#            """, unsafe_allow_html=True)
#
#     st.title("Fully Automated Spectral-Temporal-Velocity Estimation Framework of Doppler Stress Echocardiography")
#     analyzer = DopplerEnvelopeAnalyzer(
#         prf=17,
#         spectrum_region_ratio=(0.49, 1.0),
#         baseline_exclude_height=40,
#         text_region_width_ratio=0.1,
#         left_text_width_ratio=0.14,
#         left_text_height_ratio=0.65,
#         speed_scale_height_ratio=(0.43, 0.955),
#         speed_scale_width_ratio=0.07
#     )
#     param_col, result_col = st.columns([1, 2])  # 左侧占1/3，右侧占2/3
#     # 左侧参数选择栏
#     with st.sidebar:
#
#         st.markdown("""
#             <style>
#             /* 堆叠图像容器：移除固定高度，使用相对定位 */
#             .image-stack {
#                 position: relative;
#                 margin-bottom: 5px;  /* 底部留白，避免遮挡后续内容 */
#             }
#             /* 堆叠图像样式：调整偏移量，避免溢出 */
#             .stacked-img {
#                 position: absolute;
#                 border: 1px solid #ddd;
#                 box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
#             }
#             /* 标题样式：确保类名正确对应 */
#             h1.custom-heading {
#                 font-size: 24px;
#                 font-weight: 700;
#                 color: #2C3E50;  /* 深灰色文字 */
#                 margin-bottom: 20px;
#                 border-bottom: 2px solid #AAD4EC;  /* 蓝色底边框 */
#                 padding-bottom: 8px;
#             }
#             /* 处理步骤容器：确保在堆叠图像下方 */
#             .process-steps {
#                 margin-top: 300px;  /* 手动设置顶部间距（可根据实际图像高度调整） */
#                 position: relative;
#                 z-index: 10;  /* 高于堆叠图像的z-index（默认z-index为0） */
#             }
#             </style>
#
#             <!-- 标题标签：确保class名与CSS对应，且标签闭合 -->
#             <h1 class='custom-heading'>📊 Image & Process Flow</h1>
#             """, unsafe_allow_html=True)
#
#         # 1. 原始图像展示
#         # image_path = r"D:\桌面文件\jpg图片\原图\0409\IM_0358.jpg"
#         _path = r"D:\雨婷\超声多普勒分析\jpg_results\3.png"
#         folder_path = r"D:\雨婷\超声多普勒分析\jpg图片\精简\0409精简"
#
#         image_paths = [
#             r"D:\雨婷\超声多普勒分析\jpg图片\原图\0409\IM_0358.jpg",
#             r"D:\雨婷\超声多普勒分析\jpg图片\原图\0409\IM_0359.jpg",
#             r"D:\雨婷\超声多普勒分析\jpg图片\原图\0409\IM_0361.jpg",
#             r"D:\雨婷\超声多普勒分析\jpg图片\原图\0409\IM_0362.jpg"
#         ]
#
#         # 检查所有图片是否存在
#         missing_paths = [p for p in image_paths if not os.path.exists(p)]
#         if missing_paths:
#             st.error(f"❌ 缺失图片: {', '.join(missing_paths)}")
#         else:
#             # 读取所有图片并转换为base64
#             base64_images = []
#             for path in image_paths:
#                 with open(path, "rb") as f:
#                     base64_images.append(base64.b64encode(f.read()).decode("utf-8"))
#
#             # 显示标题
#             st.markdown("### Original Doppler Images")
#
#             # Create stacking container
#             col = st.columns(1)[0]
#             with col:
#                 st.markdown('<div class="image-stack">', unsafe_allow_html=True)
#
#                 # Display images in reverse order (first image at bottom)
#                 for i, img_b64 in enumerate(reversed(base64_images)):
#                     left = i * 20
#                     top = i * 15
#                     z_index = i
#
#                     st.markdown(f"""
#                             <img src="data:image/jpg;base64,{img_b64}"
#                                  class="stacked-img"
#                                  style="left: {left}px; top: {top}px; z-index: {z_index};
#                                         width: 90%; max-width: 600px; position: absolute;">
#                         """, unsafe_allow_html=True)
#
#                 st.markdown('</div>', unsafe_allow_html=True)
#
#         if os.path.exists(_path):
#             with open(_path, "rb") as img_file:
#                 img_bytes = img_file.read()
#                 img_base64 = base64.b64encode(img_bytes).decode("utf-8")
#
#                 st.markdown("""
#                                 <div class="process-steps">
#                                     <div class="image-card" style="padding-top: 5px; margin-bottom: 100px;">
#                                         <h3 style="color: #2C3E50; margin-top: 0; font-family: 'Segoe UI', sans-serif;">Processing Steps</h3>
#                                         <img src="data:image/jpg;base64,{}" style="width:100%; border-radius:5px; margin-top:10px;">
#                                     </div>
#                                 </div>
#                             """.format(img_base64), unsafe_allow_html=True)
#         else:
#             st.error(f"Processing steps image not found: {_path}")
#
#         # 处理按钮
#         process_button = st.button("🚀 Process Image")
#
#     # 检查文件夹是否存在
#     if not os.path.exists(folder_path):
#         st.error(f"文件夹路径不存在: {folder_path}")
#         return
#
#     # 处理按钮
#     if process_button:
#         with st.spinner("Processing video frame-by-frame, please wait......"):
#             image_path = r"D:\雨婷\超声多普勒分析\heatmap\val2017\50560820240722_LY-FUHE-KONGGE_20240722090753241.jpg" # 或者从UI选择
#             fig,fig1,fig2,fig3,fig4= analyzer.visualize(image_path)
#
#             if fig and fig1 and fig2 and fig3 and fig4:
#                 st.success("First frame processing completed")
#
#                 def local_img_to_base64(path):
#                     import base64
#                     with open(path, "rb") as f:
#                         return base64.b64encode(f.read()).decode()
#                 def fig_to_base64(fig):
#                     buf = BytesIO()
#                     fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
#                     buf.seek(0)
#                     return base64.b64encode(buf.getvalue()).decode("utf-8")
#
#                     # 将所有图像转换为Base64
#
#                 fig1_base64 = local_img_to_base64(r"D:\雨婷\超声多普勒分析\heatmap\D\one\6.jpg")
#                 fig2_base64 = local_img_to_base64(r"D:\雨婷\超声多普勒分析\heatmap\results新\result_valid_6.jpg")
#                 fig3_base64 =local_img_to_base64(r"D:\雨婷\超声多普勒分析\jpg_results\聚类选择.png")
#                 fig4_base64 = fig_to_base64(fig3)
#                 fig5_base64 = fig_to_base64(fig4)
#                 fig6_base64 = local_img_to_base64(r"D:\雨婷\超声多普勒分析\jpg_results\角度.png")
#                 # 创建2x2网格布局
#                 col1, col2 = st.columns(2)
#                 col3, col4 = st.columns(2)
#                 col5, col6 = st.columns(2)
#
#                 # 卡片1: QRST Complex Detection (完整HTML卡片，包含标题+图像)
#                 with col1:
#                     st.markdown(f"""
#                        <div style="
#                            background: white;
#                            border-radius: 10px;
#                            padding: 15px;
#                            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
#                            margin-bottom: 20px;
#                            border-left: 4px solid #AAD4EC;
#                        ">
#                            <h3 style="color: #2C3E50; margin-top: 0; font-family: 'Segoe UI', sans-serif;">📈 Heatmap</h3>
#                            <img src="data:image/png;base64,{fig1_base64}" style="width:100%; border-radius:5px; margin-top:10px;">
#                        </div>
#                        """, unsafe_allow_html=True)
#
#                 # 卡片2: Envelope Extraction and Smoothing
#                 with col2:
#                     st.markdown(f"""
#                        <div style="
#                            background: white;
#                            border-radius: 10px;
#                            padding: 15px;
#                            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
#                            margin-bottom: 20px;
#                            border-left: 4px solid #AAD4EC;
#                        ">
#                            <h3 style="color: #2C3E50; margin-top: 0; font-family: 'Segoe UI', sans-serif;">🔍 Detection Results</h3>
#                            <img src="data:image/png;base64,{fig2_base64}" style="width:100%; border-radius:5px; margin-top:10px;">
#                        </div>
#                        """, unsafe_allow_html=True)
#
#                 # 卡片3: Peak Positioning
#                 with col3:
#                     st.markdown(f"""
#                        <div style="
#                            background: white;
#                            border-radius: 10px;
#                            padding: 15px;
#                            padding-bottom: 100px;
#                            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
#                            margin-bottom: 20px;
#                            border-left: 4px solid #AAD4EC;
#                        ">
#                            <h3 style="color: #2C3E50; margin-top: 0; font-family: 'Segoe UI', sans-serif;">📍 Cluster Selection</h3>
#                            <img src="data:image/png;base64,{fig3_base64}" style="width:100%; border-radius:5px; margin-top:140px;">
#                        </div>
#                        """, unsafe_allow_html=True)
#
#                 # 卡片4: Waveform Completion
#                 with col4:
#                     st.markdown(f"""
#                        <div style="
#                            background: white;
#                            border-radius: 10px;
#                            padding: 15px;
#                            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
#                            margin-bottom: 20px;
#                            border-left: 4px solid #AAD4EC;
#                        ">
#                            <h3 style="color: #2C3E50; margin-top: 0; font-family: 'Segoe UI', sans-serif;">🧩 Envelope and Waveform Completion</h3>
#                            <img src="data:image/png;base64,{fig4_base64}" style="width:100%; border-radius:5px; margin-top:10px;">
#                        </div>
#                        """, unsafe_allow_html=True)
#                 with col5:
#                     st.markdown(f"""
#                                 <div style="
#                                           background: white;
#                                           border-radius: 10px;
#                                           padding: 15px;
#                                           box-shadow: 0 4px 12px rgba(0,0,0,0.08);
#                                           margin-bottom: 20px;
#                                           border-left: 4px solid #AAD4EC;
#                                       ">
#                                           <h3 style="color: #2C3E50; margin-top: 0; font-family: 'Segoe UI', sans-serif;">🔬 Parameter Extraction</h3>
#                                           <img src="data:image/png;base64,{fig5_base64}" style="width:100%; border-radius:5px; margin-top:10px;">
#                                       </div>
#                                       """, unsafe_allow_html=True)
#                     with col6:
#                         st.markdown(f"""
#                                           <div style="
#                                               background: white;
#                                               border-radius: 10px;
#                                               padding: 15px;
#                                               padding-bottom: 20px;
#                                               box-shadow: 0 4px 12px rgba(0,0,0,0.08);
#                                               margin-bottom: 20px;
#                                               border-left: 4px solid #AAD4EC;
#                                           ">
#                                               <h3 style="color: #2C3E50; margin-top: 0; font-family: 'Segoe UI', sans-serif;">⚙️Real-time Angle Correction</h3>
#                                               <img src="data:image/png;base64,{fig6_base64}" style="width:100%; border-radius:5px; margin-top:10px;">
#                                           </div>
#                                           """, unsafe_allow_html=True)
#             analyzer = BatchDopplerAnalyzer(folder_path)
#             results = analyzer.analyze_folder()
#
#             if results:
#                 st.success(f"The peak line chart processing completed.")
#
#                 # 显示结果图表
#                 fig1, fig2, fig3 ,fig4= analyzer.plot_combined_curves()
#
#                 if fig1 and fig2 and fig3 and fig4:
#                     def fig_to_base64(fig):
#                         buf = BytesIO()
#                         fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
#                         buf.seek(0)
#                         return base64.b64encode(buf.getvalue()).decode("utf-8")
#
#                     fig1_base64 = fig_to_base64(fig1)
#                     fig2_base64 = fig_to_base64(fig2)
#                     fig3_base64 = fig_to_base64(fig3)
#                     fig4_base64 = fig_to_base64(fig4)
#
#                     # 创建包含四张图的卡片
#                     st.markdown(f"""
#                             <div style="
#                                 background: white;
#                                 border-radius: 12px;
#                                 padding: 20px;
#                                 box-shadow: 0 4px 12px rgba(0,0,0,0.08);
#                                 margin-bottom: 25px;
#                                 border-left: 5px solid #AAD4EC;
#                             ">
#                             <h3 style="
#                             color: #2C3E50;
#                             margin-top: 0;
#                             margin-bottom: 15px;
#                             font-family: 'Segoe UI', sans-serif;
#                             font-weight: 600;
#                             ">Hemodynamic-Time Curve
#                             </h3>
#                                 <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px;">
#                                     <!-- 统一尺寸容器 -->
#                                     <div style="width: 100%; height: 350px; display: flex; justify-content: center; align-items: center;">
#                                         <img src="data:image/png;base64,{fig1_base64}" style="max-width:100%; max-height:330px; border-radius:5px;">
#                                     </div>
#                                     <div style="width: 100%; height: 350px; display: flex; justify-content: center; align-items: center;">
#                                         <img src="data:image/png;base64,{fig2_base64}" style="max-width:100%; max-height:330px; border-radius:5px;">
#                                     </div>
#                                     <div style="width: 100%; height: 350px; display: flex; justify-content: center; align-items: center;">
#                                         <img src="data:image/png;base64,{fig3_base64}" style="max-width:100%; max-height:330px; border-radius:5px;">
#                                     </div>
#                                     <div style="width: 100%; height: 350px; display: flex; justify-content: center; align-items: center;">
#                                         <img src="data:image/png;base64,{fig4_base64}" style="max-width:100%; max-height:330px; border-radius:5px;">
#                                     </div>
#                                 </div>
#                             </div>
#                         """, unsafe_allow_html=True)
#                 else:
#                     st.warning("未能生成完整的图表")