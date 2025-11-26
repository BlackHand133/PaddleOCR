# Thai WER (Word Error Rate) Metric

เมตริก WER สำหรับภาษาไทยที่ใช้ตัดคำด้วย 3 tokenizers:
- **newmm** - PyThaiNLP default tokenizer
- **attacut** - PyThaiNLP's attacut tokenizer
- **deepcut** - Deepcut tokenizer

## คุณสมบัติ

- ✅ ทำงานเฉพาะเมื่อเรียกผ่าน `python tools/eval.py`
- ✅ ใช้ environment แยกต่างหากสำหรับ Thai tokenizers
- ✅ ประเมิน WER จาก 3 tokenizers พร้อมกันในครั้งเดียว
- ✅ เลือกใช้ tokenizer แบบ flexible (ใช้ทั้ง 3 หรือเลือกเฉพาะบางตัว)
- ✅ คำนวณ WER เฉลี่ย (average) อัตโนมัติ
- ✅ ไม่กระทบกับ dependencies ของโปรเจคหลัก

## การติดตั้ง

### 1. สร้าง Virtual Environment แยกสำหรับ Thai Tokenizers

```bash
# สร้าง virtual environment ใหม่
python -m venv thai_tokenizer_env

# เปิดใช้งาน environment (Windows)
thai_tokenizer_env\Scripts\activate

# เปิดใช้งาน environment (Linux/Mac)
source thai_tokenizer_env/bin/activate

# ติดตั้ง dependencies
pip install -r tools/thai_tokenizer_requirements.txt
```

### 2. ทดสอบ Thai Tokenizer Script

```bash
# ใช้ Python จาก thai_tokenizer_env
thai_tokenizer_env\Scripts\python tools\thai_tokenizer.py --help
```

## การใช้งาน

### การเพิ่ม WER Metric ในไฟล์ Config

เพิ่มการตั้งค่าใน config file (เช่น `configs/rec/th_rec_config.yml`):

```yaml
Metric:
  name: ThaiWERMetric
  main_indicator: wer_newmm
  python_path: 'thai_tokenizer_env/Scripts/python'  # Windows
  # python_path: 'thai_tokenizer_env/bin/python'   # Linux/Mac
  enabled: true
```

### พารามิเตอร์

- **name**: ต้องระบุเป็น `ThaiWERMetric`
- **main_indicator**: เมตริกหลักที่จะใช้ (ค่าเริ่มต้น: `wer_avg`)
  - ตัวเลือก: `wer_avg`, `wer_newmm`, `wer_attacut`, `wer_deepcut`
  - แนะนำใช้ `wer_avg` เพื่อดูค่าเฉลี่ยจากทุก engine
- **engines**: (optional) รายการ tokenizers ที่ต้องการใช้ (ค่าเริ่มต้น: ทั้ง 3 ตัว)
  - ตัวเลือก: `['newmm', 'attacut', 'deepcut']`
  - ตัวอย่าง:
    - `engines: ['newmm', 'attacut', 'deepcut']` - ใช้ทั้ง 3 ตัว (ค่าเริ่มต้น)
    - `engines: ['newmm']` - ใช้เฉพาะ newmm
    - `engines: ['newmm', 'attacut']` - ใช้เฉพาะ newmm และ attacut
- **python_path**: path ไปยัง Python interpreter ที่ติดตั้ง Thai tokenizers
  - Windows: `thai_tokenizer_env/Scripts/python` หรือ `thai_tokenizer_env\\Scripts\\python.exe`
  - Linux/Mac: `thai_tokenizer_env/bin/python`
- **tokenizer_script**: (optional) path ไปยัง `thai_tokenizer.py`
  - ค่าเริ่มต้น: auto-detect ที่ `tools/thai_tokenizer.py`
- **enabled**: เปิด/ปิดการคำนวณ WER (ค่าเริ่มต้น: `true`)

### การรัน Evaluation

```bash
# รันจาก environment หลักของ PaddleOCR (ไม่ใช่ thai_tokenizer_env)
python tools/eval.py -c configs/rec/th_rec_config.yml
```

## ผลลัพธ์

เมตริกที่จะแสดง:
- `wer_newmm`: WER จาก newmm tokenizer
- `wer_attacut`: WER จาก attacut tokenizer
- `wer_deepcut`: WER จาก deepcut tokenizer
- `wer_avg`: WER เฉลี่ยจากทุก engine ที่เปิดใช้งาน

### ตัวอย่างผลลัพธ์

```
metric eval ***************
wer_newmm: 0.1234
wer_attacut: 0.1456
wer_deepcut: 0.1389
wer_avg: 0.1360
```

ค่า WER ที่ต่ำกว่า = ผลลัพธ์ดีกว่า (0.0 = perfect match)

## การทำงาน

### WER (Word Error Rate) คำนวณอย่างไร?

WER = (Substitutions + Insertions + Deletions) / Total words in reference

โดยใช้ Levenshtein distance ในระดับคำ (word-level)

### ขั้นตอนการทำงาน

1. **เช็คว่ารันจาก `tools/eval.py` หรือไม่**
   - ถ้าไม่ใช่จะปิดการทำงานอัตโนมัติ

2. **สำหรับแต่ละ batch:**
   - เขียน predictions และ labels ลง temporary JSON file
   - เรียก `thai_tokenizer.py` ผ่าน subprocess ด้วย Python จาก environment แยก
   - อ่านผลลัพธ์ที่ตัดคำแล้วกลับมา
   - คำนวณ WER สำหรับทั้ง 3 tokenizers

3. **รวมผลลัพธ์**
   - คำนวณค่าเฉลี่ย WER จากทุก batch
   - แสดงผลสำหรับแต่ละ tokenizer

## ข้อควรระวัง

1. **Path Separator**
   - Windows ใช้ `\\` หรือ `/` ใน path
   - Linux/Mac ใช้ `/`

2. **Performance**
   - การเรียก subprocess มี overhead
   - เหมาะสำหรับ evaluation ไม่เหมาะสำหรับ training

3. **Timeout**
   - Tokenization มี timeout 60 วินาที
   - สามารถปรับได้ใน `wer_metric.py` (ค้นหา `timeout=60`)

## Troubleshooting

### ข้อผิดพลาด: "Tokenizer script not found"

ตรวจสอบว่า `tools/thai_tokenizer.py` มีอยู่ และ path ถูกต้อง

### ข้อผิดพลาด: "pythainlp not installed"

ตรวจสอบว่าติดตั้ง dependencies ใน thai_tokenizer_env แล้ว:

```bash
thai_tokenizer_env\Scripts\python -c "import pythainlp; print(pythainlp.__version__)"
```

### ข้อผิดพลาด: "WER calculation will be skipped"

ตรวจสอบว่า:
1. `python_path` ใน config ถูกต้อง
2. Thai tokenizer environment ถูกติดตั้งครบถ้วน

### WER แสดงผล 0.0 ตลอด

ตรวจสอบว่า:
1. กำลังรันจาก `python tools/eval.py` (ไม่ใช่วิธีอื่น)
2. `enabled: true` ใน config
3. ดูใน console output ว่ามี warning message อะไร

## ตัวอย่าง Config ที่สมบูรณ์

```yaml
Global:
  use_gpu: true
  epoch_num: 100
  log_smooth_window: 20
  print_batch_step: 10
  save_model_dir: ./output/th_rec_model
  save_epoch_step: 10
  eval_batch_step: 500
  cal_metric_during_train: True
  pretrained_model:
  checkpoints:
  use_visualdl: False
  infer_img:
  character_dict_path: ./ppocr/utils/dict/th_dict.txt
  max_text_length: 80
  infer_mode: False
  use_space_char: True
  save_res_path: ./output/rec/predicts.txt

Architecture:
  model_type: rec
  algorithm: SVTR_LCNet
  # ... (architecture details)

Loss:
  name: CTCLoss

Optimizer:
  name: AdamW
  # ... (optimizer details)

# Thai WER Metric Configuration
Metric:
  name: ThaiWERMetric
  main_indicator: wer_newmm
  python_path: 'thai_tokenizer_env/Scripts/python'
  enabled: true

Train:
  dataset:
    # ... (training dataset config)

Eval:
  dataset:
    # ... (evaluation dataset config)
```

## การใช้งานร่วมกับ Metrics อื่น

ถ้าต้องการใช้ WER ร่วมกับ RecMetric สามารถใช้ได้โดย:

1. ใช้ RecMetric เป็นหลักในการ train
2. เพิ่ม ThaiWERMetric สำหรับ evaluation เท่านั้น

หรือสร้าง config แยกสำหรับ evaluation:

**train_config.yml** (ใช้ RecMetric):
```yaml
Metric:
  name: RecMetric
  main_indicator: acc
```

**eval_config.yml** (ใช้ ThaiWERMetric):
```yaml
Metric:
  name: ThaiWERMetric
  main_indicator: wer_newmm
  python_path: 'thai_tokenizer_env/Scripts/python'
```

จากนั้นรัน:
```bash
# Training
python tools/train.py -c train_config.yml

# Evaluation with WER
python tools/eval.py -c eval_config.yml
```

## License

เหมือนกับ PaddleOCR (Apache License 2.0)
