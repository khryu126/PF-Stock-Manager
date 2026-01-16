#!python3.11
import sys
import os
import subprocess
import ssl  # SSL 인증 오류 해결을 위해 추가

# [0] 초동 조치: SSL 인증서 확인 건너뛰기 및 버전 체크
def startup_setup():
    # DINOv2 모델 다운로드 시 발생하는 SSL: CERTIFICATE_VERIFY_FAILED 오류 방지
    ssl._create_default_https_context = ssl._create_unverified_context
    
    print("="*60)
    print("🚀 시스템 초기화 중... (Python 3.11 환경 확인)")
    print(f"🐍 현재 실행 엔진: Python {sys.version.split()[0]}")
    print("="*60)
    
    target = (3, 11)
    if sys.version_info[:2] != target:
        print(f"\n⚠️ 현재 버전이 3.11이 아닙니다. 3.11로 전환을 시도합니다...")
        try:
            # 절대 경로 확보 및 실행 (공백/특수문자 포함 경로 대응)
            script_path = os.path.abspath(__file__)
            subprocess.run(['py', '-3.11', script_path], check=True)
            sys.exit()
        except Exception as e:
            print(f"\n❌ 3.11 전환 실패: {e}")
            input("\n종료하려면 엔터(Enter)를 누르세요...")
            sys.exit()

# 프로그램 시작 전 환경 세팅 실행
startup_setup()

try:
    # [1] 필수 라이브러리 로드
    import pickle
    import tkinter as tk
    from tkinter import filedialog, messagebox
    import traceback
    from PIL import Image
    import numpy as np
    import torch
    import torchvision.transforms as T
    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
    from tensorflow.keras.preprocessing import image as k_image
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    Image.MAX_IMAGE_PIXELS = None
    print("✅ 라이브러리 및 AI 엔진 로드 성공!")

except Exception as e:
    print("\n" + "!"*60)
    print("❌ 라이브러리 로딩 중 에러가 발생했습니다.")
    print(f"에러 메시지: {e}")
    print("\n👉 해결: 'py -3.11 -m pip install tensorflow torch torchvision pillow numpy'를 입력하세요.")
    print("!"*60)
    input("\n종료하려면 엔터(Enter)를 누르세요...")
    sys.exit()

# [2] 메인 추출 프로세스
def run_generator():
    try:
        # 경로 인식 오류(Errno 2) 방지를 위해 작업 디렉토리 고정
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True) # 선택 창을 맨 앞으로

        print("\n📂 [1단계] 분석할 자재 사진 폴더를 선택하세요.")
        input_dir = filedialog.askdirectory(title="자재 사진 폴더 선택")
        if not input_dir: return

        print("📂 [2단계] 결과물(material_features.pkl) 저장 폴더를 선택하세요.")
        output_dir = filedialog.askdirectory(title="저장 폴더 선택")
        if not output_dir: return

        # AI 모델 빌드 (DINOv2 + ResNet50)
        print("\n" + "="*50)
        print("🤖 하이브리드 AI 엔진 구동 중 (첫 실행 시 모델 다운로드 진행)...")
        
        # ResNet50 (형태/구조 파악용)
        model_res = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        
        # DINOv2 (미세 질감/패턴 파악용)
        model_dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        model_dino.eval()
        
        dino_transform = T.Compose([
            T.Resize(224), T.CenterCrop(224), T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        print("✅ 엔진 준비 완료!")

        # 이미지 스캔 및 특징 추출
        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')
        all_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_ext)]
        
        feature_db = {}
        total = len(all_files)
        print(f"✨ 분석 시작: 총 {total}개 파일")

        for idx, fname in enumerate(all_files):
            try:
                img_path = os.path.join(input_dir, fname)
                raw_img = Image.open(img_path).convert('RGB')
                
                # ResNet50 연산
                x = k_image.img_to_array(raw_img.resize((224, 224)))
                res_vec = model_res.predict(preprocess_input(np.expand_dims(x, axis=0)), verbose=0).flatten()
                
                # DINOv2 연산
                dino_in = dino_transform(raw_img).unsqueeze(0)
                with torch.no_grad():
                    dino_vec = model_dino(dino_in).cpu().numpy().flatten()
                
                # 결합 및 용량 최적화 (float16 적용하여 25MB 제한 대응)
                feature_db[fname] = np.concatenate([res_vec, dino_vec]).astype(np.float16)

                print(f"🚀 [{idx + 1}/{total}] 완료: {fname}")
            except Exception as e:
                print(f"❌ {fname} 분석 오류 건너뜀: {e}")

        # 피클 파일 저장
        out_path = os.path.join(output_dir, 'material_features.pkl')
        with open(out_path, 'wb') as f:
            pickle.dump(feature_db, f, protocol=pickle.HIGHEST_PROTOCOL)

        final_size = os.path.getsize(out_path) / (1024 * 1024)
        print("\n" + "="*50)
        print(f"✅ 추출 성공! 최종 용량: {final_size:.2f} MB")
        messagebox.showinfo("완료", f"하이브리드 지문 생성 완료!\n용량: {final_size:.2f} MB")

    except Exception as e:
        print(f"\n❌ 작업 중 오류 발생:\n{traceback.format_exc()}")
        input("\n내용을 확인하신 후 엔터를 누르세요...")

if __name__ == "__main__":
    run_generator()
    input("\n모든 작업 완료. 종료하려면 엔터(Enter)를 누르세요...")