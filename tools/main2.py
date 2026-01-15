import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pydicom
from datetime import datetime


def is_dicom_file(file_path):
    """检查文件是否为DICOM格式"""
    try:
        with open(file_path, 'rb') as f:
            f.seek(128)
            header = f.read(4)
            return header == b'DICM'
    except Exception:
        return False


def convert_dicom_to_jpg(dicom_folder, output_folder):
    """将DICOM文件转换为JPG并添加清晰的时间日期信息"""
    os.makedirs(output_folder, exist_ok=True)

    # 尝试加载系统字体，若失败则使用默认字体
    try:
        # 优先尝试加载系统中的无衬线字体，字号加大到24以提高可读性
        if os.name == 'nt':  # Windows系统
            font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 24)  # 使用粗体Arial
        else:  # Linux或macOS系统
            font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSansBold.ttf", 24)
    except:
        # 加载默认字体并调整字号
        font = ImageFont.load_default()
        # 对于较新版本的PIL，尝试调整默认字体大小
        try:
            font = ImageFont.load_default(size=24)
        except:
            pass  # 旧版本PIL不支持size参数，保持默认

    for filename in os.listdir(dicom_folder):
        file_path = os.path.join(dicom_folder, filename)
        if os.path.isfile(file_path) and is_dicom_file(file_path):
            try:
                # 读取DICOM文件
                ds = pydicom.dcmread(file_path)

                # 获取像素数据并转换为图像
                pixel_array = ds.pixel_array

                # 处理不同位数的DICOM图像（确保正确转换为8位图像）
                if pixel_array.dtype != np.uint8:
                    # 归一化到0-255
                    pixel_array = ((pixel_array - np.min(pixel_array)) /
                                   (np.max(pixel_array) - np.min(pixel_array)) * 255).astype(np.uint8)

                image = Image.fromarray(pixel_array)

                # 获取时间日期信息
                date_time_info = []
                if hasattr(ds, 'ContentDate') and ds.ContentDate:
                    content_date = ds.ContentDate
                    # 格式化日期 (YYYYMMDD -> YYYY-MM-DD)
                    if len(content_date) >= 8:
                        formatted_date = f"{content_date[:4]}-{content_date[4:6]}-{content_date[6:8]}"
                        date_time_info.append(f"Date: {formatted_date}")

                if hasattr(ds, 'ContentTime') and ds.ContentTime:
                    content_time = ds.ContentTime
                    # 格式化时间 (HHMMSS.FFFFFF -> HH:MM:SS)
                    if len(content_time) >= 6:
                        formatted_time = f"{content_time[:2]}:{content_time[2:4]}:{content_time[4:6]}"
                        date_time_info.append(f"Time: {formatted_time}")

                # 如果有时间或日期信息，添加到图像上
                if date_time_info:
                    draw = ImageDraw.Draw(image)
                    # 将两行合并为一行，用空格分隔
                    text = "  ".join(date_time_info)  # 使用两个空格作为分隔

                    # 获取文本尺寸
                    text_bbox = draw.textbbox((0, 0), text, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]

                    # 文本位置：顶部居中，留出10像素边距
                    margin = 10
                    x = (image.width - text_width) // 2  # 水平居中
                    y = margin  # 顶部

                    # 直接绘制白色文本（不再使用半透明背景）
                    draw.text((x, y), text, font=font, fill="white")

                # 保存为JPG
                output_filename = os.path.splitext(filename)[0] + ".jpg"
                output_file_path = os.path.join(output_folder, output_filename)
                image.save(output_file_path, "JPEG", quality=95)  # 提高JPG质量
                print(f"Converted {filename} to {output_file_path}")

            except Exception as e:
                print(f"Failed to convert {filename}: {e}")


if __name__ == "__main__":
    # 请修改为您的DICOM文件夹和输出文件夹路径
    dicom_folder = 'D:\\桌面文件\\FUHE-长时间\\0626FUHE略减低'
    output_folder = 'D:\\桌面文件\\0626'
    convert_dicom_to_jpg(dicom_folder, output_folder)