import setuptools

url = "https://github.com/MERTULAS/learned"

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(name="learned",
                 version="0.5.0",
                 description="Some machine learning.",
                 long_description="It contains deep learning model, result metrics and data preprocessing, regression, classification, recognition algorithms.",
                 author="H.Mert ULAS",
                 url=url,
                 license='MIT',
                 author_email="h.mert.ulas@gmail.com",
                 packages=setuptools.find_packages(),
                 classifiers=[
                  'Intended Audience :: Developers',
                  'Intended Audience :: Education',
                  'Intended Audience :: Science/Research',
                  'License :: OSI Approved :: MIT License',
                  'Programming Language :: Python :: 3',
                  'Topic :: Software Development :: Libraries',
                  'Topic :: Software Development :: Libraries :: Python Modules'
                 ],
                 python_requires='>=3.5.0',
                 install_requires=["numpy>=1.14.0"])
