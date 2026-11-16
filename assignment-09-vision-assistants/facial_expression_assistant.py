import sys
import cv2


def analyze(image_path):
    from deepface import DeepFace
    result = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=True)
    face = result[0] if isinstance(result, list) else result
    print('Estimated facial-expression category:', face['dominant_emotion'])
    print('Model expression probabilities:', face['emotion'])
    print('Facial expressions are not reliable evidence of a person’s actual emotions or mental state.')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python facial_expression_assistant.py path/to/face.jpg')
        raise SystemExit(2)
    analyze(sys.argv[1])
