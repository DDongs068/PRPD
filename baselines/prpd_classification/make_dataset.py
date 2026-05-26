import os # os 라이브러리 호출
import re # re 라이브러리 호출
import warnings # warnings 라이브러리 호출 
warnings.filterwarnings("ignore", category=UserWarning) # 경고 메시지 무시를 위한 코드 

def extract_time_from_filename(file_name): # 파일의 이름에서 시간 정보만 추출하는 함수 정의
    match = re.search(r'(\d{6}_\d{6})', file_name) # 정규표현식을 이용하여 6자리_6자리(년월일_시분초)로 구성된 시간 정보를 파일의 이름에서 찾고 match 변수에 저장
    if match: # 6자리_6자리로 구성된 시간 정보를 찾은 경우에만 작동
        return match.group(1) # 6자리_6자리(년월일_시분초)로 구성된 시간 정보 반환
    return "" # 6자리_6자리(년월일_시분초)로 구성된 시간 정보를 찾지 못한 경우 빈 문자열을 반환

def path_maker(base_path, dataset_type): # 원천데이터 경로 반환을 위한 함수 정의
    
    dataset_type_dict = {'전체':['고체','액체','기체'], '고체':['고체'], '액체':['액체'], '기체':['기체']} # 데이터셋 생성을 위한 딕셔너리 생성
    pd_type_check_list = ['정상','노이즈','표면 방전','표면방전','보이드 방전','보이드방전','코로나 방전','코로나방전'] # 폴더 이름 중 클래스(정상, 노이즈, 표면방전, 코로나방전, 보이드방전)이 아닌 경우를 제외하기 위한 리스트 선언
    insulator_check_list = dataset_type_dict[dataset_type] # dataset_type에 따라 절연체 종류(전체, 고체, 액체, 기체)가 아닌 경우를 제외하기 위해 dataset_type_dict내 Value 선택
    equipment_check_list = ['ACSR-OC','CNCV-W','TFR-CV','7.2kV배전반','22.9kV배전반','25.8kVGIS','단상유입변압기','전력용유입변압기','계기용변압기'] # 폴더 이름 중 목표 전력설비명과 다른경우를 제외하기 위한 리스트 생성
    all_folders = [] 

    pd_type_list = os.listdir(base_path) # 원천 데이터 내 클래스(정상, 노이즈, 표면방전, 코로나방전, 보이드방전)에 대한 리스트 생성
    for pd_type in pd_type_list: # pd_type_list내 요소 만큼 순회
        if pd_type in pd_type_check_list: # pd_type이 pd_type_check_list 에 있는 경우에만 작동
            pd_type_path = base_path + pd_type + '/' # pd_type_check_list에 pd_type이 존재하는 경우 pd_type_path 생성 
            insulator_list = os.listdir(pd_type_path) # 위에서 생성한 경로를 통해 insulator의 리스트를 생성
            for insulator in insulator_list: # insulator_list내 요소만큼 순회
                if insulator in insulator_check_list: # insulator가 insulator_check_list에 존재하는 경우 작동
                    insulator_path = pd_type_path + insulator + '/' # insulator_check_list에 insulator가 존재하는 경우 insulator_path 생성
                    equipment_list = os.listdir(insulator_path) # 위에서 생성한 경로를 통해 equipment의 리스트를 생성
                    for equipment in equipment_list: # ecuipment_list내 요소만큼 순회
                        if equipment in equipment_check_list: # equipment_check_list 내 equipment가 존재하는 경우 작동
                            equipment_path = insulator_path + equipment + '/' # equipment_check_list에 equipmnent가 존재하는 경우 equipment_path를 생성
                            all_folders.append(equipment_path) # equipment_path를 all_folders 리스트에 추가
    return all_folders # 원천데이터가 존재하는 모든 폴더를 반환

def make_dataset(current_path, ratio=1, dataset_type = '전체'): # 데이터셋 생성을 위한 함수 정의

    all_folders = path_maker(current_path, dataset_type) # path_maker 함수를 통한 원천데이터 내 모든 폴더 경로 리스트 생성
    train_dataset = [] 
    valid_dataset = [] 
    test_dataset = []  

    for path in all_folders: # all_folders 내 요소만큼 순회
        
        files = [path + f for f in os.listdir(path) if f.lower().endswith('.png')] # path 내에 파일 만큼 순회하고, f가 소문자 .png로 끝날경우 path와 f를 더해 files 리스트에 추가
        
        files.sort(key=lambda x: extract_time_from_filename(os.path.basename(x))) # 시간순서대로 Train, Validation, Test, Dataset을 나누기 위해 extract_time_from_filename을 통해 시간 정보를 추출하여 시간 순서로 정렬 진행
        
        total_files = len(files) # 현재 폴더의 전체 데이터 계수 
        train_files = files[:int(total_files * 0.8 * ratio)] # 정렬된 전체 데이터 중 0.8비율 만큼 Train 데이터셋 생성
        valid_files = files[int(total_files * 0.8 * ratio):int(total_files * 0.9 * ratio)] #  Train 데이터셋을 제외한 0.2 비율 만큼의 데이터 중 0.1 만큼의 데이터를 통해  Validation 데이터셋 생성
        test_files  = files[int(total_files * 0.9 * ratio):int(total_files * ratio)] # 마지막 0.1 만큼의 데이터를 통해 Test 데이터셋 생성
            
        train_dataset.extend(train_files) # 위에서 생성한 Train Dataset을 train_dataset 리스트에 extend
        valid_dataset.extend(valid_files) # 위에서 생성한 Validtaion Dataset을 valid_dataset 리스트에 extend
        test_dataset.extend(test_files)   # 위에서 생성한 Test Dataset을 test_dataset 리스트에 extend

    return train_dataset, valid_dataset, test_dataset # 생성한 데이터셋 반환