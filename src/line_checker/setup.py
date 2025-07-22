from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'line_checker'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),  # line_checker/ 하위의 모든 파이썬 패키지 포함
    package_data={
        'line_checker': [
            'LC_resource/best.pt',
            'LC_resource/warning_banner.png',
            'LC_resource/warning_icon.png',
            'LC_resource/test_video/*',
        ],
    },
    include_package_data=True,
    data_files=[
        ('share/ament_index/resource_index/packages', ['line_checker/resource/line_checker']),
        ('share/' + package_name, ['package.xml']),
        # 기타 필요시 추가
    ],
    install_requires=[
        'setuptools',
        'rclpy',
        'opencv-python',
        'PyQt5',
        'ultralytics',
        'numpy',
        # 필요시 추가 패키지
    ],
    zip_safe=True,
    maintainer='hkit',
    maintainer_email='0708joon@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'LC = line_checker.main:main',  # LC 명령어로 main.py의 main() 실행
        ],
    },
)
