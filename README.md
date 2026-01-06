# CV-Final-Project

## 下載圖片資料集

```
wget -O 電腦視覺專案圖片.zip https://data.hongyu.dev/computer-vision-mid-project/電腦視覺專案圖片.zip 
unzip 電腦視覺專案圖片.zip -d temp_images && mv temp_images/電腦視覺專案圖片/pic/* data/pictures/ && rm -r temp_images 電腦視覺專案圖片.zip
```
## Available Models 
```
┏━━━━━━━━━━━━━━━━┓
┃ Model Name     ┃
┡━━━━━━━━━━━━━━━━┩
│ VGG16          │
│ VGG19          │
│ ResNet50       │
│ ResNet101      │
│ ResNet152      │
│ DenseNet121    │
│ DenseNet169    │
│ DenseNet201    │
│ InceptionV3    │
│ EfficientNetB6 │
└────────────────┘
```

## CLI 使用說明
```shell
# 訓練全部模型
uv run main.py train

# 訓練指定模型
uv run main.py train --model VGG16

# 基準測試全部模型
uv run main.py benchmark

# 基準測試指定模型
uv run main.py benchmark --model ResNet50

# 列出支援的模型
uv run main.py list-models
```

## 結果
訓練及基準測試結果會儲存在 `data/models/{MODEL_NAME}/` 目錄下，包含模型權重、訓練歷史及損失/準確率曲線圖。
