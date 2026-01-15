import pydicom

def print_dicom_metadata(file_path):
    ds = pydicom.dcmread(file_path)
    print("===== DICOM Metadata =====")
    for elem in ds:
        print(elem)
    print("==========================")

# 使用示例
dicom_file = "D:\\桌面文件\\原码\\10.30\\IM_0088"
print_dicom_metadata(dicom_file)