#!/bin/bash

INPUT_DIR="ucf_avi"  # 비디오 파일들이 있는 폴더
OUTPUT_DIR="ucf_avi_trimmed_2s"  # 결과를 저장할 폴더

mkdir -p "$OUTPUT_DIR"  # 출력 폴더 생성

for file in "$INPUT_DIR"/*; do
    if [[ -f "$file" ]]; then  # 파일인지 확인
        filename=$(basename "$file")
        extension="${filename##*.}"  # 파일 확장자 추출
        base_name="${filename%.*}"  # 확장자를 제외한 파일명 추출

        # FFmpeg 명령으로 10초로 자르기
        ffmpeg -i "$file" -t 2 -c copy "$OUTPUT_DIR/${base_name}.$extension"
    fi
done
