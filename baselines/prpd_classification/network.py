import torch.nn as nn # torch.nn 라이브러리를 nn이라는 이름으로 사용할수있도록 호출
import torchvision # torchvision 라이브러리 호출

class ResNet152(nn.Module): 
    def __init__(self, output_ch=5): # output_ch은 클래스를 의미하기 때문에 정상, 노이즈, 표면 방전, 코로나 방전, 보이드 방전 다섯개 클래스 학습을 위한 설정
        super(ResNet152, self).__init__() 
        
        self.resnet = torchvision.models.resnet152(pretrained=True) # torchvision에서 제공하는 ResNet152를 resnet 변수에 할당
        self.resnet.fc = nn.Sequential( # Sequential을 통해 사전학습된 resnet의 마지막 레이어를 본 학습 Task에 맞는 분류기 설정을 위해 output_ch = 5로 수정
                                        nn.Dropout(p=0.5, inplace=True), # Droptout 추가 
                                        nn.Linear(2048, 1024), # nn.linear를 통한 분류기 첫 번째 계층 추가
                                        nn.ReLU(inplace=True), # 활성함수 ReLU 추가
                                        nn.Linear(1024, output_ch) # nn.linear를 통한 분류기 두 번째 계층 추가
                                        )
    def forward(self, x): 
        return self.resnet(x) #resnet의 forward 결과 반환


class EfficientNet_b0(nn.Module): 
    def __init__(self, output_ch=5): # output_ch은 클래스를 의미하기 때문에 정상, 노이즈, 표면 방전, 코로나 방전, 보이드 방전 다섯개 클래스 학습을 위한 설정
        super(EfficientNet_b0, self).__init__() 
        
        self.efficientnet = torchvision.models.efficientnet_b0(pretrained=True) # torchvision에서 제공하는 efficientnet_b0를 efficientnet 변수에 할당
        self.efficientnet.classifier = nn.Sequential( # Sequential을 통해 사전학습된 efficientnet_b0의 마지막 레이어를 본 학습 Task에 맞는 분류기 설정을 위해 output_ch = 5로 수정
                                                    nn.Dropout(p=0.2, inplace=True),# Droptout 추가
                                                    nn.Linear(in_features=1280, out_features=512, bias=True), # nn.linear를 통한 분류기 첫 번째 계층 추가
                                                    nn.ReLU(inplace=True), # 활성함수 ReLU 추가
                                                    nn.Linear(in_features=512, out_features=output_ch, bias=True) # nn.linear를 통한 분류기 두 번째 계층 추가
                                                    )
    def forward(self, x):
        return self.efficientnet(x) # efficientnet의 forward 결과 반환