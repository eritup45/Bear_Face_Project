20200325
# Disease_Prevention_Project
(因為environment無法匯出，所以手動安裝package)
## 安裝步驟:

```powershell=
conda install -c conda-forge opencv

到https://pypi.org/simple/dlib/	下載dlib-19.8.1-cp36-cp36m-win_amd64.whl
pip install dlib-19.8.1-cp36-cp36m-win_amd64.whl

pip install face_recognition
conda install -c conda-forge psutil
conda install -c conda-forge dataclasses
pip install easygui
```

## 執行步驟：
```powershell=
cd Bear_Face_Project\Disease_Prevention_Project\V7\
```

## 執行主程式：

```powershell=
python video_demo.py [PATH]
PATH: video_demo.py的位置
```

EX.

```powershell=
python .\video_demo.py D:\大三\專題\防疫專案\Others\Bear_Face_Project\Disease_Prevention_Project\V7
```

### 注意：
V7前一個資料夾需要有Start_temp.bat
V7需要有Release資料夾

## demo流程：
1. 先選擇要當作人臉辨識的相機，選完後會執行FLIR的Start_temp.bat(只先將他們的.bat設為pause)
![](https://i.imgur.com/WsrYBnd.jpg)

2. 在框內才會偵測人臉
![](https://i.imgur.com/DxWL81d.jpg)

3. 按s儲存照片
![](https://i.imgur.com/MA0nKSC.jpg)

7. 按q退出

