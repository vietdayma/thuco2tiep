from setuptools import find_packages, setup
from typing import List
HYPEN_E_DOT='-e .'  # Chuỗi đặc biệt để xác định trường hợp cài đặt ở chế độ phát triển

def get_requirements(file_path:str)->List[str]:
    '''
    Hàm này trích xuất danh sách các thư viện phụ thuộc từ file requirements.txt
    
    Parameters:
        file_path (str): Đường dẫn đến file chứa danh sách thư viện
        
    Returns:
        List[str]: Danh sách tên các thư viện cần cài đặt
        
    Chức năng:
    - Đọc từng dòng trong file requirements.txt
    - Xử lý các dòng để loại bỏ ký tự xuống dòng
    - Loại bỏ -e. nếu có (cài đặt ở chế độ phát triển)
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n"," ") for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
    name="CO2 Emission Prediction by Vehicle",  # Tên gói phần mềm
    version=1.0,  # Phiên bản hiện tại
    author="Rajveer Singh",  # Tác giả
    packages=find_packages(),  # Tự động tìm tất cả các package trong dự án
    install_requires = get_requirements('requirements.txt')  # Cài đặt các thư viện phụ thuộc
)