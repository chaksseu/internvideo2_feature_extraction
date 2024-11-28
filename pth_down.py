from huggingface_hub import hf_hub_download

# Hugging Face Repository 정보
repo_id = "OpenGVLab/InternVideo2-Stage2_1B-224p-f4"  # 예: "pytorch/vision"
filename = "InternVideo2-stage2_1b-224p-f4.pt"  # 다운로드할 파일 이름


#/OpenGVLab/InternVideo2-Stage1-1B-224p-f8/resolve/main/pretrain.pth


#OpenGVLab/InternVideo2-Stage2_1B-224p-f4
#InternVideo2-stage2_1b-224p-f4.pt

# 다운로드
file_path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir="/home/jovyan/fileviewer/MMG/VideoEncoder/")
print(f"파일이 다운로드되었습니다: {file_path}")
