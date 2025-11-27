# เมตริกขั้นสูงสำหรับ Text Detection ใน PaddleOCR

## ภาพรวม

เอกสารนี้อธิบายเมตริกขั้นสูงสำหรับการประเมินผล text detection ใน PaddleOCR โดยเฉพาะคลาส `DetMetricAdvanced` ที่มีความสามารถในการคำนวณ AP/mAP และการประเมินผลแบบหลาย threshold สำหรับงานวิจัย

## ฟีเจอร์หลัก

### 1. เมตริกมาตรฐาน (เหมือน DetMetric)
- **Precision**: ความแม่นยำของการ detect (TP / (TP + FP))
- **Recall**: ความครอบคลุมของการ detect (TP / (TP + FN))
- **F-measure (hmean)**: ค่าเฉลี่ยฮาร์โมนิกของ precision และ recall

### 2. การประเมินผลแบบหลาย Threshold
ประเมินผลที่ระดับ IoU หลายระดับ:
- Threshold เริ่มต้น: `[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]`
- แสดงเมตริกละเอียด (Precision, Recall, F1) สำหรับแต่ละ threshold
- ช่วยให้เข้าใจประสิทธิภาพของโมเดลในระดับความแม่นยำต่างๆ

### 3. mAP (mean Average Precision)
- คำนวณค่าเฉลี่ยของ F1-score จากทุก IoU threshold
- ใช้เป็นเมตริกหลักในการเลือกโมเดลที่ดีที่สุด (เมื่อตั้ง `main_indicator: mAP`)
- เป็นเมตริกมาตรฐานที่ใช้ในงานวิจัย object detection สมัยใหม่

## วิธีการใช้งาน

### 1. การตั้งค่า Config

สร้างหรือแก้ไขไฟล์ config สำหรับ detection ให้ใช้ `DetMetricAdvanced`:

```yaml
Metric:
  name: DetMetricAdvanced
  main_indicator: mAP
  iou_thresholds: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
```

**พารามิเตอร์:**
- `name`: ต้องเป็น `DetMetricAdvanced`
- `main_indicator`: เมตริกที่ใช้เลือกโมเดลที่ดีที่สุด ตัวเลือก: `mAP`, `hmean`, `precision`, `recall` (ค่าเริ่มต้น: `mAP`)
- `iou_thresholds`: รายการของ IoU threshold ที่จะประเมิน (ค่าเริ่มต้น: 10 threshold จาก 0.5 ถึง 0.95)
- `iou_constraint`: IoU threshold สำหรับการจับคู่ prediction กับ ground truth (ค่าเริ่มต้น: 0.5)
- `area_precision_constraint`: อัตราส่วนพื้นที่ทับซ้อนสำหรับ "don't care" regions (ค่าเริ่มต้น: 0.5)

### 2. ตัวอย่างการตั้งค่า

#### ตัวอย่างที่ 1: การประเมินแบบเต็มรูปแบบ
```yaml
Metric:
  name: DetMetricAdvanced
  main_indicator: mAP
  iou_thresholds: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
```

#### ตัวอย่างที่ 2: แบบ COCO-style AP@[.5:.95]
```yaml
Metric:
  name: DetMetricAdvanced
  main_indicator: mAP
  iou_thresholds: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
```

#### ตัวอย่างที่ 3: จำกัด Threshold (ประเมินเร็วขึ้น)
```yaml
Metric:
  name: DetMetricAdvanced
  main_indicator: mAP
  iou_thresholds: [0.5, 0.7, 0.9]
```

#### ตัวอย่างที่ 4: มาตรฐาน + AP@0.75
```yaml
Metric:
  name: DetMetricAdvanced
  main_indicator: hmean
  iou_thresholds: [0.5, 0.75]
```

### 3. การรันการประเมินผล

ใช้คำสั่งประเมินผลมาตรฐาน:

```bash
python tools/eval.py -c configs/det/det_mv3_db_advanced.yml
```

### 4. รูปแบบผลลัพธ์

ผลลัพธ์จะแสดง:

```
precision: 0.8523
recall: 0.7891
hmean: 0.8194
mAP: 0.7856
IoU@0.50: P:0.8523 R:0.7891 F1:0.8194
IoU@0.55: P:0.8401 R:0.7802 F1:0.8090
IoU@0.60: P:0.8267 R:0.7698 F1:0.7972
IoU@0.65: P:0.8120 R:0.7580 F1:0.7841
IoU@0.70: P:0.7965 R:0.7450 F1:0.7698
IoU@0.75: P:0.7798 R:0.7307 F1:0.7544
IoU@0.80: P:0.7612 R:0.7150 F1:0.7374
IoU@0.85: P:0.7410 R:0.6980 F1:0.7189
IoU@0.90: P:0.7189 R:0.6795 F1:0.6987
IoU@0.95: P:0.6950 R:0.6595 F1:0.6768
```

**คำอธิบาย:**
- `precision`, `recall`, `hmean`: เมตริกมาตรฐานที่ IoU=0.5
- `mAP`: ค่าเฉลี่ยของ F1-score จากทุก threshold
- `IoU@X.XX`: เมตริกละเอียดที่แต่ละ IoU threshold
  - `P`: Precision
  - `R`: Recall
  - `F1`: F-measure (hmean)

## เปรียบเทียบกับเมตริกที่มีอยู่

| คลาสเมตริก | Precision/Recall | Multi-threshold | mAP | การใช้งาน |
|------------|------------------|-----------------|-----|-----------|
| `DetMetric` | ✅ (IoU=0.5) | ❌ | ❌ | ประเมินมาตรฐาน, เร็ว |
| `DetFCEMetric` | ✅ (confidence thresholds) | ✅ (แบบ confidence) | ❌ | โมเดล FCE/DRRG |
| `DetMetricAdvanced` | ✅ (IoU=0.5) | ✅ (แบบ IoU) | ✅ | งานวิจัย, เผยแพร่ |

## เมื่อไหร่ควรใช้ DetMetricAdvanced

### ✅ แนะนำสำหรับ:
1. **งานวิจัย** ที่ต้องการเมตริกครอบคลุม
2. **เปรียบเทียบโมเดล** ในระดับความแม่นยำต่างๆ
3. **เผยแพร่งานวิจัย** ใน conference/journal ชั้นนำ
4. **Ablation studies** วิเคราะห์คุณภาพการระบุตำแหน่ง
5. **การส่งแข่งขัน** ที่ต้องการ mAP metrics

### ⚠️ พิจารณาใช้ DetMetric มาตรฐานสำหรับ:
1. **การทดลองรวดเร็ว** และการพัฒนา
2. **ติดตามการ train** (ประเมินช้ากว่าเล็กน้อย)
3. **Production deployment** (threshold เดียวเพียงพอ)
4. **ทรัพยากรคอมพิวเตอร์จำกัด**

## ข้อพิจารณาด้านประสิทธิภาพ

- **เวลาประเมิน**: ช้ากว่า `DetMetric` ประมาณ 10 เท่า เนื่องจากประเมินหลาย threshold
- **การใช้หน่วยความจำ**: ใกล้เคียงกับ `DetMetric`
- **ผลกระทบต่อ training**: ตั้งค่า `cal_metric_during_train: False` (แนะนำ)
- **คำแนะนำ**: ใช้ `eval_batch_step` ควบคุมความถี่ในการประเมิน

## การผสานรวมกับ Config ที่มีอยู่

แปลง config ที่มีอยู่ให้ใช้เมตริกขั้นสูง:

1. เปลี่ยนส่วน Metric:
```yaml
# ก่อน
Metric:
  name: DetMetric
  main_indicator: hmean

# หลัง
Metric:
  name: DetMetricAdvanced
  main_indicator: mAP
```

2. ปรับความถี่การประเมินถ้าจำเป็น:
```yaml
Global:
  eval_batch_step: [0, 5000]  # ประเมินถี่น้อยลงเพราะเมตริกช้ากว่า
```

## ตัวอย่าง Config แบบเต็ม

ดูตัวอย่างที่ `configs/det/det_mv3_db_advanced.yml`

## รายละเอียดทางเทคนิค

### การจับคู่ด้วย IoU Threshold
- Prediction จะถูกจับคู่กับ ground truth โดยใช้ greedy matching
- คำนวณ IoU โดยใช้ Shapely polygon intersection/union
- แต่ละ ground truth จับคู่กับ prediction ได้มากสุด 1 ตัว
- แต่ละ prediction จับคู่กับ ground truth ได้มากสุด 1 ตัว

### การคำนวณ mAP
```python
# สำหรับแต่ละ IoU threshold t:
#   คำนวณ precision_t, recall_t, f1_t
#
# mAP = mean(f1_0.5, f1_0.55, ..., f1_0.95)
```

### Confidence Scores
- ถ้า detection มี confidence scores จะถูกเก็บไว้แต่ยังไม่ได้ใช้ในการคำนวณ AP
- เวอร์ชันอนาคตอาจรวมการคำนวณ AP แบบเต็มรูปแบบด้วย PR curves

## เอกสารอ้างอิง

1. ICDAR 2015 Text Detection Challenge: โปรโตคอลการประเมินมาตรฐาน
2. COCO Detection Challenge: วิธีการคำนวณ mAP แบบหลาย threshold
3. PaddleOCR DB: Differentiable Binarization สำหรับ text detection

## คำถามที่พบบ่อย

**Q: สามารถใช้ DetMetricAdvanced ระหว่าง training ได้หรือไม่?**
A: ได้ แต่ควรตั้งค่า `cal_metric_during_train: False` และปรับ `eval_batch_step` ให้ประเมินถี่น้อยลง เพราะใช้เวลานาน

**Q: DetMetricAdvanced backward compatible กับ DetMetric หรือไม่?**
A: ใช่ มันให้เมตริกมาตรฐานทั้งหมด (precision, recall, hmean ที่ IoU=0.5) บวกกับเมตริกหลาย threshold

**Q: จะเลือกโมเดลที่ดีที่สุดอย่างไร?**
A: ตั้งค่า `main_indicator` ใน config ตัวเลือกที่นิยม: `mAP` (ครอบคลุม), `hmean` (มาตรฐาน)

**Q: สามารถกำหนด IoU thresholds เองได้หรือไม่?**
A: ได้ ตั้งค่า `iou_thresholds: [0.5, 0.6, 0.7]` หรือรายการ threshold ใดๆ ใน config

**Q: ใช้ได้กับทุก detection algorithm หรือไม่?**
A: ได้ ใช้ได้กับ DB, EAST, PSE, SAST, FCE และ detection algorithm ทั้งหมดใน PaddleOCR

## การสนับสนุน

หากมีปัญหาหรือคำถาม กรุณาเปิด issue ที่ PaddleOCR GitHub repository
