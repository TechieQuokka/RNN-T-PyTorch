# Vanish CLI 명령어 가이드

오디오 노이즈 제거를 위한 명령줄 인터페이스 완벽 가이드

---

## 목차

1. [기본 명령어](#기본-명령어)
2. [단일 파일 처리](#단일-파일-처리)
3. [일괄 처리](#일괄-처리)
4. [설정 파일](#설정-파일)
5. [시스템 정보](#시스템-정보)
6. [실전 예제](#실전-예제)

---

## 기본 명령어

### 도움말
```bash
vanish --help              # 전체 명령어 도움말
vanish process --help      # process 명령어 옵션
vanish batch --help        # batch 명령어 옵션
vanish --version           # 버전 확인
```

---

## 단일 파일 처리

### 기본 사용법

```bash
# 가장 간단한 사용
vanish process input.wav

# 출력 파일 지정
vanish process input.wav -o output.wav
vanish process input.wav --output clean_audio.wav
```

**기본 동작:**
- 입력: `input.wav`
- 출력: `input_clean.wav` (자동 생성)
- 품질: balanced
- 향상: resemble
- 메트릭: 표시

---

### 품질 프리셋

```bash
# 빠른 처리 (30초/1분 오디오)
vanish process audio.wav -q fast
vanish process audio.wav --quality fast

# 균형 처리 (60초/1분 오디오) [기본값]
vanish process audio.wav -q balanced
vanish process audio.wav --quality balanced

# 고품질 처리 (120초/1분 오디오)
vanish process audio.wav -q high
vanish process audio.wav --quality high
```

**품질 프리셋 비교:**

| 프리셋 | Demucs Shifts | Resemble Steps | 처리 시간 | 품질 |
|--------|---------------|----------------|----------|------|
| fast | 0 | 15 | ~30초 | 보통 |
| balanced | 1 | 30 | ~60초 | 좋음 |
| high | 2 | 50 | ~120초 | 우수 |

---

### 향상 모델 선택

```bash
# Resemble-Enhance 사용 [기본값]
vanish process audio.wav --enhancement resemble

# VoiceFixer 사용
vanish process audio.wav --enhancement voicefixer

# 둘 다 사용 (fallback)
vanish process audio.wav --enhancement both
```

**모델 선택 가이드:**
- `resemble`: 최신 AI 모델, 고품질 (기본 추천)
- `voicefixer`: 빠른 처리, 안정적
- `both`: resemble 실패 시 voicefixer로 fallback

---

### 처리 장치 선택

```bash
# GPU 사용 (자동 감지)
vanish process audio.wav

# GPU 강제 사용
vanish process audio.wav -d cuda
vanish process audio.wav --device cuda

# CPU 사용
vanish process audio.wav -d cpu
vanish process audio.wav --device cpu
```

**GPU 권장:**
- RTX 3060 이상
- CUDA 11.8+
- CPU보다 5-10배 빠름

---

### 메트릭 제어

```bash
# 메트릭 표시 [기본값]
vanish process audio.wav --metrics

# 메트릭 숨기기 (빠른 처리)
vanish process audio.wav --no-metrics
```

**출력되는 메트릭:**
- SNR (Signal-to-Noise Ratio) - dB
- PESQ (음성 품질) - 1.0~4.5
- STOI (명료도) - 0.0~1.0
- Spectral Convergence

---

### 중간 파일 저장

```bash
# 중간 파일 저장
vanish process audio.wav --save-intermediate

# 저장 안 함 [기본값]
vanish process audio.wav --no-save-intermediate
```

**저장되는 중간 파일:**
1. `01_preprocessed.wav` - 전처리된 오디오
2. `02_vocals_separated.wav` - 분리된 보컬
3. `02_background.wav` - 분리된 배경
4. `03_enhanced.wav` - 향상된 음성
5. `04_postprocessed.wav` - 후처리 완료

**용도:** 디버깅, 품질 확인, 중간 단계 분석

---

### 로그 레벨

```bash
# 경고만 표시 [기본값]
vanish process audio.wav

# 진행 상황 표시
vanish process audio.wav -v
vanish process audio.wav --verbose

# 상세 정보 표시
vanish process audio.wav -vv

# 디버그 정보 표시
vanish process audio.wav -vvv
```

**로그 레벨:**
- (기본): WARNING - 경고와 에러만
- `-v`: INFO - 진행 상황 포함
- `-vv`: DEBUG - 상세 디버그
- `-vvv`: DEBUG + 서브모듈 로그

---

### 설정 파일 사용

```bash
# 설정 파일로 처리
vanish process audio.wav -c config.yaml
vanish process audio.wav --config my_config.yaml
```

**설정 파일 우선순위:**
1. CLI 옵션 (최우선)
2. 설정 파일
3. 기본값

---

### 복합 옵션

```bash
# 고품질 + GPU + 중간파일 저장 + 상세로그
vanish process audio.wav \
  -q high \
  -d cuda \
  --save-intermediate \
  -vv

# 빠른 처리 + VoiceFixer + 메트릭 숨김
vanish process audio.wav \
  -q fast \
  --enhancement voicefixer \
  --no-metrics

# 설정 파일 + 출력 지정
vanish process audio.wav \
  -c custom_config.yaml \
  -o output/clean.wav \
  -v
```

---

## 일괄 처리

### 기본 사용법

```bash
# 디렉토리 내 모든 WAV 파일 처리
vanish batch input_dir/ output_dir/

# 현재 디렉토리 처리
vanish batch . output/
```

**기본 동작:**
- 패턴: `*.wav`
- 출력 접두사: `clean_`
- 재귀: 하위 디렉토리 포함 안 함

---

### 파일 패턴

```bash
# WAV 파일만 [기본값]
vanish batch input/ output/ --pattern "*.wav"

# MP3 파일만
vanish batch input/ output/ --pattern "*.mp3"

# 모든 오디오 파일
vanish batch input/ output/ --pattern "*.{wav,mp3,flac}"

# 특정 이름 패턴
vanish batch input/ output/ --pattern "voice_*.wav"
```

---

### 출력 파일명

```bash
# 기본 접두사 "clean_"
vanish batch input/ output/
# input/audio.wav → output/clean_audio.wav

# 커스텀 접두사
vanish batch input/ output/ --prefix "processed_"
# input/audio.wav → output/processed_audio.wav

# 접두사 없음
vanish batch input/ output/ --prefix ""
# input/audio.wav → output/audio.wav
```

---

### 설정 파일 사용

```bash
# 설정 파일로 일괄 처리
vanish batch input/ output/ -c config.yaml

# 설정 + 패턴 + 접두사
vanish batch input/ output/ \
  -c config.yaml \
  --pattern "*.mp3" \
  --prefix "cleaned_"
```

---

### 로그 레벨

```bash
# 진행 상황 표시
vanish batch input/ output/ -v

# 상세 로그
vanish batch input/ output/ -vv
```

---

### 실전 예제

```bash
# 10개 파일 빠르게 처리
vanish batch recordings/ output/ \
  --pattern "*.wav" \
  --prefix "clean_" \
  -v

# MP3 파일 고품질 처리
vanish batch podcast/ output/ \
  --pattern "*.mp3" \
  -c high_quality.yaml \
  -vv

# 대용량 일괄 처리 (메트릭 숨김)
vanish batch large_dataset/ output/ \
  --no-metrics \
  -v
```

---

## 설정 파일

### 설정 파일 생성

```bash
# 기본 설정 생성
vanish create-config config.yaml

# RTX 3060 최적화 [추천]
vanish create-config config.yaml --preset rtx3060

# 빠른 처리 프리셋
vanish create-config config.yaml --preset fast

# 고품질 프리셋
vanish create-config config.yaml --preset high_quality
```

---

### 프리셋 설명

**rtx3060** (기본 추천)
```yaml
# RTX 3060 12GB 최적화
demucs:
  shifts: 1
  batch_size: 1
  segment: 10
resemble:
  denoiser_run_steps: 30
  enhance_run_steps: 30
  chunk_size: 441000
```

**fast** (빠른 처리)
```yaml
# 처리 속도 우선
demucs:
  shifts: 0
resemble:
  denoiser_run_steps: 15
  enhance_run_steps: 15
```

**high_quality** (최고 품질)
```yaml
# 품질 우선
demucs:
  shifts: 2
resemble:
  denoiser_run_steps: 50
  enhance_run_steps: 50
```

---

### 설정 파일 편집

생성된 `config.yaml` 수동 편집:

```yaml
# 장치 설정
device: cuda

# 오디오 설정
target_sr: 44100
channels: 1
enhancement_mode: resemble

# 출력 설정
output_format: wav
output_bitdepth: 16
save_intermediate: false

# Demucs 설정
demucs:
  model: htdemucs
  device: cuda
  shifts: 1
  overlap: 0.25
  segment: 10

# Resemble-Enhance 설정
resemble:
  device: cuda
  denoiser_run_steps: 30
  enhance_run_steps: 30
  solver: midpoint
  nfe: 64

# 후처리 설정
postprocess:
  noise_gate_threshold: -40.0
  target_lufs: -16.0
  highpass_cutoff: 80.0
  apply_deessing: true

# 품질 평가 설정
quality:
  calculate_snr: true
  calculate_pesq: true
  calculate_stoi: true
```

---

### 설정 파일 사용

```bash
# 설정 파일로 처리
vanish process audio.wav -c config.yaml

# 설정 파일 + CLI 옵션 (CLI 우선)
vanish process audio.wav -c config.yaml -d cpu

# 일괄 처리에 설정 파일 적용
vanish batch input/ output/ -c config.yaml
```

---

## 시스템 정보

### 시스템 및 모델 확인

```bash
vanish info
```

**출력 예시:**
```
Vanish - Audio Noise Removal System

Python: 3.11.5
PyTorch: 2.1.0
CUDA available: True
CUDA version: 11.8
GPU: NVIDIA GeForce RTX 3060
GPU Memory: 12.0 GB

Model availability:
  ✅ Demucs: 4.0.1
  ✅ Resemble-Enhance: Installed
  ⚠️  VoiceFixer: Not installed (optional)
```

**용도:**
- 시스템 호환성 확인
- GPU 사용 가능 여부 확인
- 설치된 모델 확인
- 문제 해결

---

## 실전 예제

### 1. 빠른 단일 파일 처리
```bash
# 기본 설정으로 빠르게
vanish process interview.wav
```

### 2. 고품질 팟캐스트 처리
```bash
# 고품질 + 중간파일 저장 + 상세로그
vanish process podcast_ep01.wav \
  -q high \
  -o output/podcast_clean.wav \
  --save-intermediate \
  -vv
```

### 3. 대량 인터뷰 녹음 처리
```bash
# 10개 파일 일괄 처리
vanish batch interviews/ output/ \
  --pattern "interview_*.wav" \
  --prefix "clean_" \
  -v
```

### 4. CPU 환경에서 처리
```bash
# GPU 없는 환경
vanish process audio.wav \
  -d cpu \
  -q fast \
  --enhancement voicefixer
```

### 5. 커스텀 설정으로 처리
```bash
# 1. 설정 파일 생성
vanish create-config my_config.yaml --preset high_quality

# 2. 설정 수정 (선택)
nano my_config.yaml

# 3. 설정으로 처리
vanish process audio.wav -c my_config.yaml
```

### 6. MP3 파일 일괄 처리
```bash
vanish batch music/ output/ \
  --pattern "*.mp3" \
  --prefix "vocal_" \
  -q balanced \
  -v
```

### 7. 디버깅 모드
```bash
# 모든 정보 출력 + 중간파일 저장
vanish process problem_audio.wav \
  --save-intermediate \
  -vvv
```

### 8. 메트릭 비교 테스트
```bash
# 빠른 처리
vanish process test.wav -q fast -o fast_output.wav

# 고품질 처리
vanish process test.wav -q high -o high_output.wav

# 메트릭 비교
```

### 9. 프로덕션 배포
```bash
# 최적화 설정 생성
vanish create-config production.yaml --preset rtx3060

# 대량 처리
vanish batch production_data/ output/ \
  -c production.yaml \
  --pattern "*.wav" \
  --no-metrics \
  -v
```

### 10. 오디오북 처리
```bash
# 챕터별 일괄 처리
vanish batch audiobook_chapters/ clean_chapters/ \
  --pattern "chapter_*.wav" \
  --prefix "clean_" \
  -q balanced \
  -c audiobook_config.yaml \
  -v
```

---

## 명령어 치트시트

### 필수 명령어
```bash
# 기본 처리
vanish process input.wav

# 출력 지정
vanish process input.wav -o output.wav

# 품질 선택
vanish process input.wav -q fast|balanced|high

# 일괄 처리
vanish batch input_dir/ output_dir/

# 시스템 정보
vanish info

# 설정 생성
vanish create-config config.yaml --preset rtx3060
```

### 유용한 옵션
```bash
-v, -vv, -vvv           # 로그 레벨
-d cuda|cpu             # 장치 선택
-c config.yaml          # 설정 파일
-q fast|balanced|high   # 품질 프리셋
--enhancement resemble|voicefixer|both
--save-intermediate     # 중간파일 저장
--no-metrics           # 메트릭 숨김
--pattern "*.mp3"      # 파일 패턴
--prefix "clean_"      # 출력 접두사
```

---

## 문제 해결

### CUDA 메모리 부족
```bash
# CPU로 전환
vanish process audio.wav -d cpu

# 또는 빠른 프리셋
vanish process audio.wav -q fast
```

### 처리 실패 시
```bash
# 디버그 로그 + 중간파일 확인
vanish process audio.wav \
  --save-intermediate \
  -vvv
```

### 모델 설치 확인
```bash
# 시스템 정보 확인
vanish info

# 누락된 모델 설치
pip install demucs resemble-enhance voicefixer
```

### 설정 파일 오류
```bash
# 새 설정 파일 생성
vanish create-config new_config.yaml --preset rtx3060

# YAML 문법 확인
python -c "import yaml; yaml.safe_load(open('config.yaml'))"
```

---

## 성능 팁

### GPU 최적화
```bash
# RTX 3060 최적 설정
vanish create-config gpu_config.yaml --preset rtx3060
vanish process audio.wav -c gpu_config.yaml
```

### 배치 처리 최적화
```bash
# 메트릭 비활성화로 속도 향상
vanish batch input/ output/ --no-metrics -v
```

### 품질 vs 속도
```bash
# 속도 우선: fast (3배 빠름)
vanish process audio.wav -q fast

# 품질 우선: high (2배 느림, 최고 품질)
vanish process audio.wav -q high
```

---

## 추가 자료

- **전체 API 문서**: [API.md](API.md)
- **빠른 시작**: [QUICKSTART.md](QUICKSTART.md)
- **아키텍처**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **성능 가이드**: [PERFORMANCE.md](PERFORMANCE.md)
