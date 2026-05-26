import torch # torch 라이브러리 호출
import torch.nn.functional as F # torchvision.transforms 를 F 라는 이름으로 사용가능하도록 호출
from tqdm.auto import tqdm # tqdm.auto 라이브러리 내 tqdm 함수 호출
from module import * # module 스크립트 내 모든 함수 호출
import datetime # datetime 라이브러리 호출

test_loss_list = [] 
test_acc = [] 

class Start(object): 
	def __init__(self, config, test_loader):

		self.test_loader = test_loader # test를 위한 데이터 로더 선언
		self.batch_size = config.batch_size # 모델 학습을 위한 배치사이즈 설정
		self.start_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") # 실험 결과 기록을 위한 학습 시작 시간 저장을 위한 변수 선언
		self.model_type = config.model_type # Resnet/Efficientnet과 같이 모델 유형 설정을 위한 변수 선언
		self.load_model_path = config.load_model_path # 가중치 로드를 위한 사전학습된 모델 경로 선언
		self.num_classes = config.num_classes # 클래스 개수 선언 (정상, 노이즈, 표면 방전, 코로나 방전, 보이드 방전)
		self.device = config.device # GPU 지정을 위한 변수 선언
		self.model = build_model(config, mode = 'test', result_path = '', device = self.device) # model_type에 해당하는 모델 로드 및 GPU 업로드 후 self.model 변수에 인공지능 모델 할당
		self.img_ch = config.img_ch # img_channel 선언
		self.output_ch = config.output_ch # output_channel 선언
		

	def start(self): # 모델 test결과에 대한 혼동행렬 저장을 위한 함수 선언
		
		if self.load_model_path != '': # 가중치 로드를 위한 사전학습된 모델 경로가 있다면 작동
			self.model.load_state_dict(torch.load(self.load_model_path,map_location='cuda:0')) # 사전학습된 모델 경로를 통한 모델 가중치 로드 및 모델에 적용
			print('%s is Successfully Loaded from %s'%(self.model_type,self.load_model_path)) # 모델 로드 여부 터미널에 출력
	
		#===================================== Test ====================================#
		self.model.train(False) # 모델 학습 모드 종료
		self.model.eval() # 모델 평가 모드 시작

		with torch.no_grad(): # 모델 검증을 위한 기울기 갱신 기능 종료
			all_predictions = [] 
			all_labels = [] 
			raw_names = [] 
			all_probability = []
			for i, (images, label, raw_name) in enumerate(tqdm(self.test_loader,desc="[Test]")): # Test Loop

				images = images.to(self.device) # Test Dataset의 이미지를 Batch_Size 만큼 GPU에 업로드
				label = label.to(self.device) # 업로드된 이미지와 매칭되는 정답정보(정상, 노이즈, 표면 방전, 코로나 방전, 보이드 방전)를 GPU에 업로드
				result = self.model(images) # 입력 이미지에 대한 모델 예측 결과 result 변수에 저장
				all_predictions.extend(F.softmax(result, dim=1).argmax(dim=1).detach().cpu().numpy()) # 혼동 행렬 추출을 위한 예측값 계산 및 all_predictions에 추가 
				all_labels.extend(label.detach().cpu().numpy()) # 혼동 행렬 추출을 위한 모든 정답값 all_labels 리스트에 추가
				raw_names.extend(raw_name) # 이미지 데이터와 매칭되는 원본 파일명 raw_names 리스트에 추가
				all_probability.extend(F.softmax(result,dim=1).cpu().numpy()) # 예측 확률을 all_probability 리스트에 추가
		with open('/code_with_dataset/PRPD_Classification/{}_혼동행렬.csv'.format(self.model_type), 'w', encoding='euc-kr') as f:  # 혼동 행렬 저장을 위한 csv 파일 로드 
			f.write('Class ID,정상/정상,정상/노이즈,정상/표면방전,정상/코로나방전,정상/보이드방전,노이즈/정상,노이즈/노이즈,노이즈/표면방전,노이즈/코로나방전,노이즈/보이드방전,표면방전/정상,표면방전/노이즈,표면방전/표면방전,표면방전/코로나방전,표면방전/보이드방전,코로나방전/정상,코로나방전/노이즈,코로나방전/표면방전,코로나방전/코로나방전,코로나방전/보이드방전,보이드방전/정상,보이드방전/노이즈,보이드방전/표면방전,보이드방전/코로나방전,보이드방전/보이드방전\n') # csv파일에 헤더 추가
			for i in tqdm(range(len(all_labels))): # all_labels 리스트 길이 만큼 반복
				cm = confusion_matrix([all_labels[i]], [all_predictions[i]], labels=[0,1,2,3,4]) # confusion_matrix 함수를 통한 혼동행렬 생성
				f.write(raw_names[i].split('/')[-1][0:-4]+',') # raw_names의 슬라이싱을 통해 파일명 맨 마지막 일부만 csv파일에 추가
				f.write(','.join(map(str, cm.flatten()))+'\n') # csv파일에 혼동행렬 값 추가를 위해 리스트 차원 변환 및 쉼표로 구분
		with open('/code_with_dataset/PRPD_Classification/{}_모델예측값.csv'.format(self.model_type), 'w', encoding='euc-kr') as f: # 모델 예측값 저장을 위한 csv파일 로드
			f.write('모델예측값,정상,노이즈,표면방전,코로나방전,보이드방전\n') # csv파일 헤더 추가
			for i in tqdm(range(len(all_probability))): # all_probability 리스트 길이 만큼 순회
				f.write(raw_names[i].split('/')[-1][0:-4]+',') # raw_names의 슬라이싱을 통해 파일명 맨 마지막 일부만 csv파일에 추가
				f.write(','.join(map(str, list(all_probability[i])))+'\n') #  csv파일에 예측값 저장
		acc,  SE, PC, F1 = 0,0,0,0 # 정확도, 민감도, 정밀도, F1-Score 연산을 위한 변수 초기화
		acc,  SE, PC, F1  = evaluate(acc, SE, PC, F1, all_predictions, all_labels) # 정확도, 민감도, 정밀도, F1-Score 연산 및 갱신
		print('[Test] Acc: %.4f, SE: %.4f, PC: %.4f, F1: %.4f'%(acc,SE,PC,F1)+'\n{}'.format('-'*100)) # Test 성능 지표 터미널 출력