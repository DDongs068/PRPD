import torch  # torch 라이브러리 호출
import matplotlib.pyplot as plt  # matplotlib.pyplot 라이브러리를 plt라는 이름으로 사용가능하도록 호출
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score # sklearn.metrics 라이브러리 내 confusion_matrix, accuracy_score, recall_score, precision_score, f1_score 함수 호출
import seaborn as sns # seaborn 라이브러리를 sns라는 이름으로 사용가능하도록 호출
import pandas as pd # pandas 라이브러리를 pd라는 이름으로 사용가능하도록 호출

def visualize_weight_distribution(net, save_path): # 모델 가중치 분포 시각화 함수 생성
    layers_weights = [] # 모델 계층별 가중치 저장을 위한 리스트 생성  
    for m in net.modules(): # 각 계층 마다 반복 
        classname = m.__class__.__name__ # 계층 이름 저장
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1): # 계층 이름에 conv 또는 Linear 찾기 
            layers_weights.append(m.weight.data.view(-1).cpu().numpy()) # 계층 가중치 저장

    fig, axs = plt.subplots(len(layers_weights), 1, figsize=(10, 2*len(layers_weights))) # 계층 가중치 시각화를 위한 subplot활용
    for i, layer_weights in enumerate(layers_weights): # 계층 시각화를 위해 저장되어있는 만큼 반복
        axs[i].hist(layer_weights, bins=50) # 가증치 분포 시각화
        axs[i].set_title(f'Layer {i+1}') # 시각화 된 Title 작성

    plt.tight_layout() # plot 사이즈 조정 
    plt.savefig(save_path) # Plot 저장
    plt.close(fig)  # plot 닫기
       
def build_model(config, mode, result_path, device): # 모델 생성 함수
    if config.model_type == "Resnet" : # model_type이 Resnet일 경우 작동 
        from network import ResNet152 # ResNet152 클래스 호출
        model = ResNet152(output_ch=config.output_ch)
    elif config.model_type == "Efficientnet" : # model_type이 Efficientnet일 경우 작동 
        from network import EfficientNet_b0 # EfficientNet_b0 클래스 호출 
        model = EfficientNet_b0(output_ch=config.output_ch)
        
    if mode == 'train' : # mode가 train일 경우 작동 
        visualize_weight_distribution(model, result_path+'Weights_Distribution'+'.png') # visualize_weight_distribution 함수를 통한 가중치 분포 시각화 결과 저장
        
    if torch.cuda.is_available() == True : # GPU가 사용가능할때 작동 
        model.to(device) # model GPU 할당
    else: # GPU가 사용불가능할때 작동 
        model.to('cpu') # model CPU 할당
        
    return model # model반환

def evaluate(acc, SE, PC, F1, result, label): # 성능지표 계산을 위한 함수 선언
    acc = accuracy_score(label, result)  # 정확도 계산
    SE = recall_score(label, result, average = 'weighted') # 민감도 계산
    PC = precision_score(label, result, average = 'weighted') # 정밀도 계산
    F1 = f1_score(label, result, average = 'weighted') # F1-score 계산
    return acc, SE, PC, F1 # 성능지표 결과 반환

def save_confusion_matrix(y_true, y_pred, model_name, result_path): # 혼동행렬 저장을 위한 함수 선언
    label_names = ['normal','noise','surface','corona','void'] # 클래스 이름 리스트 생성
    conf_matrix = confusion_matrix(y_true, y_pred) # confusion_matrix 함수를 사용하여 conf_matrix변수에 저장

    df_cm = pd.DataFrame(conf_matrix, index = label_names, columns = label_names) # conf_matrix 리스트를 Dataframe 형태로 변환하여 df_cm 변수에 할당
    plt.figure(figsize = (10,7)) # 혼동 행렬 시각화를 위한 그래프 사이즈 설정
    sns.heatmap(df_cm, annot=True, fmt="d") # seaborn.heatmap을 활용하여 데이터프레임 시각화 

    # Add labels to the x-axis and the y-axis.
    plt.xlabel('Predicted',fontsize = 14) # x축 라벨 "Predicted"로 지정하고 폰트 사이즈=14
    plt.ylabel('True',fontsize = 14)  # y축 라벨 "True"로 지정하고 폰트 사이즈=14
    plt.savefig(result_path + f'{model_name}.png') # 혼동행렬을 저장함. result_path 위치에 {model_name}.png로 저장

    plt.close() #plot 종료