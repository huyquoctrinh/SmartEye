import  os
from time import sleep

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="gcp_cred.json"
    
def detect_text(imgPath):
    """Detects text in the file."""
    from google.cloud import vision
    import io
    client = vision.ImageAnnotatorClient()

    with io.open(imgPath, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations

    res = ""
    
    for text in texts:
        # if text == "":
        #     continue
        res += text.description + " "
    return res


img_path  = "d.jpg"
res = detect_text(img_path)
print(res)