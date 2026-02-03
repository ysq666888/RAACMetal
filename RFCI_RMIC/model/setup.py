from setuptools import setup, find_packages
from irap.Version import version

setup(
    name='irap',
    version=version,
    description='Intelligent RAAC-PSSM Protein Prediction Package',
    url='https://github.com/KingoftheNight/IRAP',
    author='Liang YC',
    author_email='1694822092@qq.com',
    license='BSD 2-Clause',
    packages=find_packages(),
    
    # 明确指定要包含的数据文件夹
    package_data={
        'irap': [
            'aaindexDB/*',
            'bin/*',
            'blastDB/*',
            'raacDB/*',
            'pdbaa',  
        ],
    },
    
    # 启用 package_data
    include_package_data=True,
    
    install_requires=['pyecharts', 'numpy', 'scikit-learn', 'pandas', 'seaborn', 'ray', 'tqdm', 'xgboost'],
    entry_points={
        'console_scripts': [
            'irap=irap.__main__:irap',
        ]
    },
    python_requires=">=3.6",
    zip_safe=False,  # 由于包含数据文件，设为 False
)