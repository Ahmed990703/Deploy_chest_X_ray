from torchvision import transforms,models
import pickle
import torch
from PIL import Image
from torch import nn





    
def inference(img):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.conv1=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc=nn.Linear(in_features=512, out_features=2, bias=True)
    model.load_state_dict(torch.load('models/model.pth'))
    with open("chest_xray/lbl_encoder.pkl","rb") as f :
        lbl_encoder=pickle.load(f)
    img_gray=img.convert("L")
    img_trans=transforms.ToTensor()(img_gray).unsqueeze(0).to(device)
    model.to(device)
    model.eval()
    preds=model(img_trans)
    clas_index=torch.argmax(preds).item()
    final_pred=lbl_encoder.inverse_transform([clas_index])[0]
    return final_pred


img_path=r'chest_xray/resized/resizedtest_NORMAL_NORMAL2-IM-0326-0001.jpeg'
img=Image.open(img_path)
preds=inference(img)
print(preds)