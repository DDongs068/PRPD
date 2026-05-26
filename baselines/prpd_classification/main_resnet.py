import os # os 라이브러리 호출
import torch # torch 라이브러리 호출
import random # random 라이브러리 호출
import argparse # argparse 라이브러리 호출
import numpy as np # numpy 라이브러리를 np라는 이름으로 사용가능하도록 호출
from torch.backends import cudnn # torch.backends 라이브러리 내 cudnn 함수 호출
from make_dataset import make_dataset # make_dataset 스크립트 내  make_dataset 함수 호출

def set_seed(seed): # 일관성 있는 학습 결과를 위해 시드 고정 함수 선언
    torch.manual_seed(seed) # 일관성 있는 학습 결과를 위해 난수 생성시 동일한 난수를 생성하기 위한 시드 고정
    torch.cuda.manual_seed_all(seed) # GPU를 통해 학습을 진행하기에 GPU에서 생성되는 난수또한 동일하게 시드 고정
    torch.backends.cudnn.deterministic = True # 합성곱 연산시 동일한 결과 도출을 위한 결정론적 알고리즘 모드 설정
    torch.backends.cudnn.benchmark = False # 합성곱 연산시 동일한 알고리즘을 통한 결과 도출을 위한 설정 
    np.random.seed(seed) # 넘파이 라이브러리 난수 시드 고정 설정
    random.seed(seed) #  random 라이브러리 난수 시드 고정 설정
    os.environ['PYTHONHASHSEED'] = str(seed) # 파이썬 해시 알고리즘의 난수 시드 고정 설정

def main(config): # 모델 학습을 위한 main함수 선언
    set_seed(42) # 시드 고정을 위한 함수 실행
    if config.mode == 'train': # config.mode가 train일 경우에 작동

        from data_loader import get_loader # 데이터 로더 호출 
        train_dataset, valid_dataset, test_dataset = make_dataset(config.raw_data_path, config.dataset_ratio, config.dataset_type) # make_dataset 함수를 통한  train_dataset, valid_dataset, test_dataset 할당
        train_loader = get_loader( # train_loader 선언
                                Data_set=train_dataset,# get_loader함수의 Dataset 변수에 traindataset 할당
                                batch_size=config.batch_size, # get_loader함수의 batch_size 변수에 config.batch_size 할당
                                num_workers=config.num_workers, # getloader함수의 numworkers 변수에 config.numworkers 할당
                                mode='train' # getloader함수의 mode 변수에 config.mode 할당
                                )
        valid_loader = get_loader( # valid_loader 선언
                                Data_set=valid_dataset, # get_loader함수의 Dataset 변수에 valid_dataset 할당
                                batch_size=config.batch_size, # get_loader함수의 batch_size 변수에 config.batch_size 할당
                                num_workers=config.num_workers, # getloader함수의 numworkers 변수에 config.numworkers 할당
                                mode='valid'# getloader함수의 mode 변수에 valid 할당
                                )
        test_loader = get_loader( # test_loader 선언
                                Data_set=test_dataset, # get_loader함수의 Dataset 변수에 test_dataset 할당
                                batch_size=config.batch_size, # get_loader함수의 batch_size 변수에 config.batch_size 할당
                                num_workers=config.num_workers, # getloader함수의 numworkers 변수에 config.numworkers 할당
                                mode='test' # getloader함수의 mode 변수에 test 할당
                                )
        from train import Train # Train 클래스 호출
            
        train = Train(config, train_loader, valid_loader, test_loader)
        train.train() # 모델 학습 시작
        
if __name__ == '__main__':  # 현재 파일이 메인 프로그램일 경우 작동
    
    parser = argparse.ArgumentParser() # 모델 학습을 위한 하이퍼 파라미터 설정
    parser.add_argument('--img_ch', type=int, default=3) # ResNet 모델의 img_channel 입력
    parser.add_argument('--output_ch', type=int, default=5) # ResNet 모델의 output_channel 입력
    parser.add_argument('--num_epochs', type=int, default=100) # ResNet 모델 학습을 위한 Epoch 입력
    parser.add_argument('--num_epochs_decay', type=int, default=70) # ResNet 모델의 Learning Rate 설정을 위한 num_epochs_decay 입력
    parser.add_argument('--num_classes', type=int, default=5) # 분류를 위한 클래스 입력
    parser.add_argument('--batch_size', type=int, default=64) # ResNet 모델 학습을 위한 batch_size 입력
    parser.add_argument('--num_workers', type=int, default=20) # 데이터 로더 설정을 위한 num_workers 입력
    parser.add_argument('--lr', type=float, default=0.0001) # ResNet 모델 학습에 사용할 Learning Rate 입력
    parser.add_argument('--beta1', type=float, default=0.5) # ResNet 모델 학습에 사용할 beta1 입력   
    parser.add_argument('--beta2', type=float, default=0.999) # ResNet 모델 학습에 사용할 beta2 입력 
    parser.add_argument('--patience', type=int, default=7) # ResNet 모델 학습에 사용할 Early Stop Patience 입력 
    parser.add_argument('--model_type', type=str, default='Resnet', help='Resnet/Efficientnet') # 모델 유형 입력(Resnet/Efficientnet)
    parser.add_argument('--model_path', type=str, default='/code_with_dataset/PRPD_Classification/ResNet_Result') # 학습 결과 저장 경로 입력
    parser.add_argument('--load_model_path', type=str, default='') # 사전학습 가중치 이용 시 모델을 불러오기 위한 경로 입력 사전학습 가중치를 사용하지 않는다면 '' 입력
    parser.add_argument('--dataset_ratio',type=float, default=1) # 데이터셋 구축을 위한 dataset_ratio 입력
    parser.add_argument('--raw_data_path', type=str, default='/code_with_dataset/원천데이터/') # 데이터셋 구축을 위한 원본 데이터셋 경로 입력
    parser.add_argument('--result_path', type=str, default='/code_with_dataset/PRPD_Classification/ResNet_Result/') # 모델 학습 결과 저장 경로 입력
    parser.add_argument('--mode', type=str, default='train', help='train') # 모델의 학습/평가 모드 입력
    parser.add_argument('--dataset_type', type=str, default='전체', help='전체/고체/액체/기체')  # 모델 학습에 사용할 데이터셋 절연체 종류 입력(전체/고체/액체/기체)
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda:0/cuda:1') # 학습에 사용할 Device 입력
    config = parser.parse_args() # config 변수에 현재까지 작성된 parser저장
    main(config) # 메인 함수 config를 기준으로 실행