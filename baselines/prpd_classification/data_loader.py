import os # os 라이브러리 호출
import random # random 라이브러리 호출
import numpy as np #numpy 라이브러리를 np라는 이름으로 사용가능하도록 호출
import torch #torch 라이브러리 호출
from torch.utils import data # 데이터 로더를 정의하기 위해 torch.utils 내 data 함수 호출
from torchvision import transforms as T # trochvision 이라는 라이브러리 내  transforms 함수를 T라는 이름으로 사용가능하도록 호출
from PIL import Image #이미지 로드를 위한 PIL라이브러리 내 Image 호출 
import pandas as pd # pandas 라이브러리를 pd 라는 이름사용가능하도록 호출


class PRPD_Classification(data.Dataset): 
	def __init__(self, Data_set, mode='train'): 
		"""Initializes image paths and preprocessing module."""
		self.image_paths = Data_set # make_dataset에서 반환한 각 데이터셋을 self.image_paths 변수에 저장
		self.mode = mode # mode(train, valid, test) self.mode에 저장
		self.set_seed(42) #시드 고정을 위한 함수 실행
		print("image count in {} path :{}".format(self.mode,len(self.image_paths))) # 현재 데이터셋의 총 개수 출력

	def __getitem__(self, index): # 데이터 매칭을 위한  __getitem__ 매서드 선언
		"""Reads an image from a file and preprocesses it and returns."""

		data_path = self.image_paths[index] # 현재 index의 이미지 데이터 경로 추출 
		image = Image.open(data_path).convert('RGB') # 이미지 데이터 load
		 
		transform = T.Compose([ # 이미지 학습을 위해 T.Compose 메소드 정의 및 transform 이라는 변수에 할당
			T.ToTensor(), # 이미지를 텐서 형태로 변환
			T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #ImageNet을 통해 추출된 채널 평균 및 표준편차를 활용하여 정규화 진행
			T.Resize((256,256)) # 이미지 크기 재조정
		])

		image = transform(image) # 이미지 전처리 및 Tensor 변환
		label_path = data_path.replace('원천데이터','라벨링데이터').replace('.png','.json') # 이미지 데이터와 매칭되는 라벨링 데이터 경로 생성
		label = pd.read_json(label_path, encoding='utf-8-sig')  # 라벨링 데이터 Load
		label = int(label['label']['PD_type']) # 라벨링 데이터 내 부분방전 유형을 불러와 저장
		label = torch.tensor(label) # 라벨 Tensor 변환
		return image, label, self.image_paths[index] # 이미지, 부분방전 유형, 이미지 경로 반환

	def set_seed(self,seed): # 일관성 있는 학습 결과를 위해 시드 고정 함수 선언
		torch.manual_seed(seed) # 일관성 있는 학습 결과를 위해 난수 생성시 동일한 난수를 생성하기 위한 시드 고정
		torch.cuda.manual_seed_all(seed) # GPU를 통해 학습을 진행하기에 GPU에서 생성되는 난수또한 동일하게 시드 고정
		torch.backends.cudnn.deterministic = True # 합성곱 연산시 동일한 결과 도출을 위한 결정론적 알고리즘 모드 설정
		torch.backends.cudnn.benchmark = False # 합성곱 연산시 동일한 알고리즘을 통한 결과 도출을 위한 설정 
		np.random.seed(seed) # 넘파이 라이브러리 난수 시드 고정 설정
		random.seed(seed) #  random 라이브러리 난수 시드 고정 설정
		os.environ['PYTHONHASHSEED'] = str(seed) # 파이썬 해시 알고리즘의 난수 시드 고정 설정
  
	def __len__(self):	# 데이터 길이 메소드 정의
		"""Returns the total number of font files."""
		return len(self.image_paths) # 데이터 길이 반환 

def get_loader( Data_set, batch_size, num_workers=2, mode='train' ): # 데이터로더 생성을 위한 함수 선언
	"""Builds and returns Dataloader."""
	
	dataset = PRPD_Classification(Data_set = Data_set, mode=mode) # 데이터셋 생성

	#위에서 생성한 데이터셋을 통한 데이터 로더 정의
	data_loader = data.DataLoader( 
									dataset=dataset, # dataset 변수에 dataset 할당
									batch_size=batch_size, # batch_size 변수에 batch_size 할당
									shuffle=True, # 데이터를 섞어서 학습에 사용할 것인지에 대한 shuffle 인자에 True 할당
									num_workers=num_workers # num_workers 변수에 num_workers 할당
									)
	return data_loader # 데이터 로더 반환