from pathlib import Path
import re
import sys
import cv2
import pytesseract


def extract_label(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(image_path)
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayscale = cv2.resize(grayscale, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    grayscale = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    text = pytesseract.image_to_string(grayscale)
    fields = {'recognized_text': text.strip()}
    date_match = re.findall(r'(?i)\b(?:EXP|Expiry|Expires)[:\s-]*([0-9/.-]+)', text)
    if date_match:
        fields['printed_expiry_text'] = date_match
    return fields

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python medicine_label_reader.py path/to/label.jpg')
        raise SystemExit(2)
    result = extract_label(Path(sys.argv[1]))
    for key, value in result.items():
        print(key, value)
    print('OCR may be incorrect. Verify all text on the original label. This tool does not recommend medicines or dosages.')
