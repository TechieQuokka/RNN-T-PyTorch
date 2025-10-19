# Vanish API 문서

AI 기반 오디오 노이즈 제거 시스템의 전체 API 레퍼런스

## 목차

1. [핵심 클래스](#핵심-클래스)
2. [설정 관리](#설정-관리)
3. [모듈 API](#모듈-api)
4. [CLI 인터페이스](#cli-인터페이스)
5. [데이터 구조](#데이터-구조)

---

## 핵심 클래스

### AudioPipeline

오디오 노이즈 제거 파이프라인의 메인 클래스

**초기화**
```python
from vanish import AudioPipeline, PipelineConfig

# 기본 설정
pipeline = AudioPipeline()

# 커스텀 설정
config = PipelineConfig(device="cuda", enhancement_mode="resemble")
pipeline = AudioPipeline(config)
```

**주요 메서드**

#### `process(input_path, output_path, save_intermediate=None) -> ProcessingResult`
단일 오디오 파일 처리

**파라미터:**
- `input_path` (str): 입력 오디오 파일 경로
- `output_path` (str): 출력 오디오 파일 경로
- `save_intermediate` (bool, optional): 중간 파일 저장 여부

**반환값:** `ProcessingResult` 객체

**처리 단계:**
1. 오디오 입력 및 전처리
2. 음원 분리 (Demucs)
3. 음성 향상 (Resemble-Enhance/VoiceFixer)
4. 후처리
5. 품질 평가

**예제:**
```python
result = pipeline.process(
    input_path="noisy_audio.wav",
    output_path="clean_audio.wav",
    save_intermediate=True
)

print(f"처리 시간: {result.processing_time:.2f}초")
print(f"SNR: {result.metrics.snr:.2f} dB")
```

#### `process_batch(input_files, output_dir, prefix="clean_") -> list[ProcessingResult]`
여러 오디오 파일 일괄 처리

**파라미터:**
- `input_files` (list[str]): 입력 파일 경로 리스트
- `output_dir` (str): 출력 디렉토리
- `prefix` (str): 출력 파일명 접두사

**반환값:** `ProcessingResult` 객체 리스트

**예제:**
```python
results = pipeline.process_batch(
    input_files=["audio1.wav", "audio2.wav"],
    output_dir="output/",
    prefix="cleaned_"
)

print(f"성공: {len(results)}/{len(input_files)}개")
```

---

## 설정 관리

### PipelineConfig

파이프라인 전체 설정을 관리하는 데이터 클래스

**기본 설정**
```python
config = PipelineConfig(
    device="cuda",              # 처리 장치 (cuda/cpu)
    target_sr=44100,            # 목표 샘플레이트 (Hz)
    channels=1,                 # 채널 수 (모노)
    enhancement_mode="resemble", # 향상 모드
    output_bitdepth=16,         # 출력 비트 깊이
    save_intermediate=False     # 중간 파일 저장
)
```

**향상 모드:**
- `"resemble"`: Resemble-Enhance 사용 (기본)
- `"voicefixer"`: VoiceFixer 사용
- `"both"`: 둘 다 사용 (fallback)

**YAML 설정 파일**

#### `from_yaml(config_path) -> PipelineConfig` (클래스 메서드)
YAML 파일에서 설정 로드

```python
config = PipelineConfig.from_yaml("config.yaml")
pipeline = AudioPipeline(config)
```

#### `to_yaml(output_path)` (인스턴스 메서드)
현재 설정을 YAML 파일로 저장

```python
config = PipelineConfig()
config.to_yaml("my_config.yaml")
```

#### `validate()`
설정 유효성 검사

```python
config = PipelineConfig(device="cuda")
config.validate()  # CUDA 사용 가능 여부 등 검증
```

### 하위 설정 클래스

#### DemucsConfig
음원 분리 설정
```python
demucs = DemucsConfig(
    model="htdemucs",     # 모델명
    device="cuda",        # 장치
    shifts=1,             # 증강 시프트 수
    overlap=0.25,         # 청크 오버랩
    segment=10,           # 세그먼트 길이(초)
    batch_size=1,         # 배치 크기
    num_workers=4         # 워커 수
)
```

#### ResembleConfig
Resemble-Enhance 설정
```python
resemble = ResembleConfig(
    device="cuda",
    denoiser_run_steps=30,  # 노이즈 제거 단계
    enhance_run_steps=30,   # 향상 단계
    solver="midpoint",      # ODE 솔버
    nfe=64,                 # 함수 평가 횟수
    chunk_size=441000       # 청크 크기 (샘플)
)
```

#### VoiceFixerConfig
VoiceFixer 설정
```python
voicefixer = VoiceFixerConfig(
    mode=2,    # 0:24kHz, 1:44.1kHz, 2:auto
    cuda=True  # CUDA 사용 여부
)
```

#### PostProcessConfig
후처리 설정
```python
postprocess = PostProcessConfig(
    noise_gate_threshold=-40.0,  # 노이즈 게이트 임계값(dB)
    target_lufs=-16.0,           # 목표 음량(LUFS)
    highpass_cutoff=80.0,        # 하이패스 필터(Hz)
    apply_deessing=True,         # 디에싱 적용
    deess_frequency=6000.0,      # 디에싱 주파수(Hz)
    deess_threshold=-20.0        # 디에싱 임계값(dB)
)
```

#### QualityConfig
품질 평가 설정
```python
quality = QualityConfig(
    calculate_snr=True,   # SNR 계산
    calculate_pesq=True,  # PESQ 계산
    calculate_stoi=True,  # STOI 계산
    target_snr=20.0,      # 목표 SNR(dB)
    target_pesq=3.5,      # 목표 PESQ
    target_stoi=0.9       # 목표 STOI
)
```

---

## 모듈 API

### AudioInput

오디오 파일 로딩 및 전처리

**초기화**
```python
from vanish.modules import AudioInput

audio_input = AudioInput(target_sr=44100)
```

**주요 메서드**

#### `preprocess(file_path, convert_mono=True, normalize_sr=True, normalize_amp=True)`
전체 전처리 파이프라인

**반환값:** `(audio, sr, metrics)` 튜플
- `audio`: 처리된 오디오 배열
- `sr`: 샘플레이트
- `metrics`: `AudioQualityMetrics` 객체

```python
audio, sr, metrics = audio_input.preprocess("input.wav")
print(f"길이: {metrics.duration:.2f}초")
print(f"RMS: {metrics.rms_level:.4f}")
```

#### `load_audio(file_path) -> (audio, sr)`
오디오 파일 로드

**지원 포맷:** WAV, MP3, FLAC, M4A/AAC, OGG

#### `convert_to_mono(audio) -> audio`
스테레오를 모노로 변환

#### `normalize_sample_rate(audio, orig_sr, target_sr=None) -> audio`
샘플레이트 정규화

#### `normalize_amplitude(audio, target_peak=0.95) -> audio`
진폭 정규화

---

### SourceSeparator

Demucs를 사용한 음원 분리

**초기화**
```python
from vanish.modules import SourceSeparator

separator = SourceSeparator(
    model_name="htdemucs",
    device="cuda",
    shifts=1,
    overlap=0.25,
    segment=10
)
```

**주요 메서드**

#### `separate(audio, sr) -> SeparatedAudio`
보컬과 배경 분리

**반환값:** `SeparatedAudio` 객체
- `vocals`: 분리된 보컬
- `background`: 분리된 배경
- `separation_quality`: 분리 품질 (0.0-1.0)

```python
separated = separator.separate(audio, sr)
print(f"분리 품질: {separated.separation_quality:.3f}")

# 보컬만 사용
vocals = separated.vocals
```

#### `load_model()`
Demucs 모델 로드 (자동 호출됨)

#### `get_model_info() -> dict`
모델 정보 조회

---

### VoiceEnhancer

음성 품질 향상 (Resemble-Enhance/VoiceFixer)

**초기화**
```python
from vanish.modules import VoiceEnhancer

# Resemble-Enhance
enhancer = VoiceEnhancer(
    model_type="resemble",
    device="cuda",
    denoiser_run_steps=30,
    enhance_run_steps=30,
    solver="midpoint",
    nfe=64
)

# VoiceFixer
enhancer_vf = VoiceEnhancer(
    model_type="voicefixer",
    device="cuda",
    mode=2,
    cuda=True
)
```

**주요 메서드**

#### `enhance_speech(vocals, sr, output_path=None) -> enhanced_audio`
음성 품질 향상

```python
enhanced = enhancer.enhance_speech(vocals, sr)
```

#### `load_model()`
향상 모델 로드 (자동 호출됨)

---

### PostProcessor

후처리 (노이즈 게이트, 정규화, 필터링)

**초기화**
```python
from vanish.modules import PostProcessor

processor = PostProcessor(
    noise_gate_threshold=-40.0,
    target_lufs=-16.0,
    highpass_cutoff=80.0,
    apply_deessing=True,
    deess_frequency=6000.0,
    deess_threshold=-20.0
)
```

**주요 메서드**

#### `process(audio, sr, apply_noise_gate=True, apply_normalization=True, apply_highpass=True, apply_deess=True) -> audio`
전체 후처리 파이프라인

```python
processed = processor.process(audio, sr)
```

**개별 처리 메서드:**

#### `apply_noise_gate(audio, threshold=None, sr=44100) -> audio`
노이즈 게이트 적용 (침묵 제거)

#### `normalize_loudness(audio, sr, target_lufs=None) -> audio`
음량 정규화 (LUFS 기준)

#### `apply_highpass_filter(audio, sr, cutoff=None) -> audio`
하이패스 필터 (저주파 제거)

#### `apply_deessing(audio, sr, frequency=None, threshold=None) -> audio`
디에싱 (치찰음 감소)

---

### QualityAssessor

품질 평가 및 메트릭 계산

**초기화**
```python
from vanish.modules import QualityAssessor

assessor = QualityAssessor(
    calculate_snr=True,
    calculate_pesq=True,
    calculate_stoi=True
)
```

**주요 메서드**

#### `assess(clean, noisy, sr) -> QualityMetrics`
전체 품질 평가

```python
metrics = assessor.assess(processed_audio, original_audio, sr)

print(f"SNR: {metrics.snr:.2f} dB")
print(f"PESQ: {metrics.pesq:.2f}")
print(f"STOI: {metrics.stoi:.3f}")

# 목표 달성 여부
if metrics.meets_targets(target_snr=20.0):
    print("품질 목표 달성!")
```

**개별 메트릭 메서드:**

#### `calculate_signal_to_noise_ratio(clean, noisy) -> float`
SNR 계산 (dB)

#### `calculate_pesq_score(reference, degraded, sr) -> float`
PESQ 점수 (1.0-4.5)

#### `calculate_stoi_score(clean, noisy, sr) -> float`
STOI 점수 (0.0-1.0)

#### `calculate_spectral_convergence(reference, processed) -> float`
스펙트럼 수렴도

---

## CLI 인터페이스

### 기본 명령어

#### 단일 파일 처리
```bash
vanish process input.wav -o output.wav

# 품질 프리셋
vanish process input.wav -q fast        # 빠른 처리
vanish process input.wav -q balanced    # 균형 (기본)
vanish process input.wav -q high        # 고품질

# 향상 모델 선택
vanish process input.wav --enhancement resemble
vanish process input.wav --enhancement voicefixer
vanish process input.wav --enhancement both

# 중간 파일 저장
vanish process input.wav --save-intermediate

# 상세 로그
vanish process input.wav -vv
```

#### 일괄 처리
```bash
vanish batch input_dir/ output_dir/

# 파일 패턴 지정
vanish batch input_dir/ output_dir/ --pattern "*.mp3"

# 출력 파일명 접두사
vanish batch input_dir/ output_dir/ --prefix "cleaned_"
```

#### 설정 파일 생성
```bash
# 프리셋으로 설정 파일 생성
vanish create-config config.yaml --preset rtx3060
vanish create-config config.yaml --preset fast
vanish create-config config.yaml --preset high_quality

# 설정 파일 사용
vanish process input.wav -c config.yaml
```

#### 시스템 정보
```bash
vanish info
```

### CLI 옵션

**공통 옵션:**
- `-o, --output PATH`: 출력 파일 경로
- `-c, --config PATH`: 설정 YAML 파일
- `-d, --device cuda|cpu`: 처리 장치
- `-v, -vv, -vvv`: 로그 레벨 (WARNING/INFO/DEBUG)

**process 명령어 옵션:**
- `-q, --quality fast|balanced|high`: 품질 프리셋
- `--enhancement resemble|voicefixer|both`: 향상 모델
- `--metrics/--no-metrics`: 메트릭 계산 여부
- `--save-intermediate/--no-save-intermediate`: 중간 파일 저장

**batch 명령어 옵션:**
- `--pattern PATTERN`: 파일 패턴 (기본: *.wav)
- `--prefix PREFIX`: 출력 파일명 접두사 (기본: clean_)

---

## 데이터 구조

### ProcessingResult

처리 결과 컨테이너

**필드:**
```python
@dataclass
class ProcessingResult:
    output_path: str              # 출력 파일 경로
    metrics: QualityMetrics       # 품질 메트릭
    processing_time: float        # 처리 시간(초)
    intermediate_files: dict      # 중간 파일 경로
```

### QualityMetrics

품질 평가 메트릭

**필드:**
```python
@dataclass
class QualityMetrics:
    snr: float                    # Signal-to-Noise Ratio (dB)
    pesq: float                   # PESQ 점수 (1.0-4.5)
    stoi: float                   # STOI 점수 (0.0-1.0)
    spectral_convergence: float   # 스펙트럼 수렴도
```

**메서드:**
- `meets_targets(target_snr, target_pesq, target_stoi)`: 목표 달성 여부
- `__str__()`: 읽기 쉬운 문자열 표현

### AudioQualityMetrics

오디오 품질 사전 평가

**필드:**
```python
@dataclass
class AudioQualityMetrics:
    duration: float          # 길이(초)
    sample_rate: int         # 샘플레이트
    channels: int            # 채널 수
    bit_depth: int           # 비트 깊이
    rms_level: float         # RMS 레벨
    dynamic_range: float     # 다이나믹 레인지(dB)
    dc_offset: float         # DC 오프셋
    clipping_detected: bool  # 클리핑 감지 여부
```

### SeparatedAudio

분리된 오디오 컨테이너

**필드:**
```python
@dataclass
class SeparatedAudio:
    vocals: np.ndarray           # 분리된 보컬
    background: np.ndarray       # 분리된 배경
    separation_quality: float    # 분리 품질 (0.0-1.0)
```

---

## 사용 예제

### 기본 사용법
```python
from vanish import AudioPipeline, PipelineConfig

# 파이프라인 생성
config = PipelineConfig(device="cuda", enhancement_mode="resemble")
pipeline = AudioPipeline(config)

# 오디오 처리
result = pipeline.process("noisy.wav", "clean.wav")

# 결과 확인
print(f"처리 완료: {result.output_path}")
print(f"소요 시간: {result.processing_time:.2f}초")
print(result.metrics)
```

### 커스텀 설정
```python
from vanish import AudioPipeline, PipelineConfig
from vanish.config import DemucsConfig, ResembleConfig

# 상세 설정
demucs = DemucsConfig(
    model="htdemucs",
    shifts=2,  # 더 높은 품질
    overlap=0.5
)

resemble = ResembleConfig(
    denoiser_run_steps=50,
    enhance_run_steps=50
)

config = PipelineConfig(
    target_sr=48000,
    enhancement_mode="resemble",
    demucs=demucs,
    resemble=resemble
)

pipeline = AudioPipeline(config)
result = pipeline.process("input.wav", "output.wav")
```

### 일괄 처리
```python
import glob

# 파일 목록 수집
audio_files = glob.glob("input_dir/*.wav")

# 일괄 처리
results = pipeline.process_batch(
    input_files=audio_files,
    output_dir="output_dir/",
    prefix="clean_"
)

# 통계 계산
avg_snr = sum(r.metrics.snr for r in results) / len(results)
print(f"평균 SNR: {avg_snr:.2f} dB")
```

### 중간 단계 접근
```python
from vanish.modules import AudioInput, SourceSeparator, VoiceEnhancer

# 개별 모듈 사용
audio_input = AudioInput(target_sr=44100)
separator = SourceSeparator(device="cuda")
enhancer = VoiceEnhancer(model_type="resemble", device="cuda")

# 1단계: 오디오 로드
audio, sr, metrics = audio_input.preprocess("input.wav")

# 2단계: 음원 분리
separated = separator.separate(audio, sr)

# 3단계: 보컬 향상
enhanced = enhancer.enhance_speech(separated.vocals, sr)

# 결과 저장
import soundfile as sf
sf.write("output.wav", enhanced, sr)
```

### 품질 평가만 수행
```python
from vanish.modules import QualityAssessor
import soundfile as sf

# 오디오 로드
clean, sr = sf.read("clean.wav")
noisy, _ = sf.read("noisy.wav")

# 품질 평가
assessor = QualityAssessor()
metrics = assessor.assess(clean, noisy, sr)

print(metrics)

# 목표 달성 여부
if metrics.meets_targets(target_snr=25.0, target_pesq=4.0):
    print("우수한 품질!")
```

---

## 의존성 요구사항

### 필수
- Python ≥ 3.9
- PyTorch ≥ 2.0
- numpy
- soundfile
- librosa
- click (CLI)

### 선택적
- demucs (음원 분리)
- resemble-enhance (음성 향상)
- voicefixer (대체 향상)
- pesq (PESQ 메트릭)
- pystoi (STOI 메트릭)
- pyloudnorm (음량 정규화)

---

## 성능 최적화

### GPU 메모리 관리 (RTX 3060 12GB)
```python
config = PipelineConfig(
    device="cuda",
    demucs=DemucsConfig(
        batch_size=1,
        segment=10,        # 10초 세그먼트
        num_workers=4
    ),
    resemble=ResembleConfig(
        chunk_size=441000  # 10초 청크
    )
)
```

### 품질 프리셋

**빠른 처리 (Fast)**
- Demucs shifts: 0
- Resemble steps: 15
- 처리 시간: ~30초 (1분 오디오)

**균형 (Balanced, 기본)**
- Demucs shifts: 1
- Resemble steps: 30
- 처리 시간: ~60초 (1분 오디오)

**고품질 (High)**
- Demucs shifts: 2
- Resemble steps: 50
- 처리 시간: ~120초 (1분 오디오)

---

## 에러 처리

### 일반적인 에러

**FileNotFoundError**
```python
try:
    result = pipeline.process("missing.wav", "output.wav")
except FileNotFoundError as e:
    print(f"파일을 찾을 수 없습니다: {e}")
```

**RuntimeError (GPU 메모리 부족)**
```python
try:
    pipeline = AudioPipeline(config)
except RuntimeError as e:
    print(f"초기화 실패: {e}")
    # CPU로 폴백
    config.device = "cpu"
    pipeline = AudioPipeline(config)
```

**ImportError (선택적 의존성)**
```python
try:
    enhancer = VoiceEnhancer(model_type="resemble")
except ImportError:
    print("Resemble-Enhance가 설치되지 않음")
    enhancer = VoiceEnhancer(model_type="voicefixer")
```

---

## 로깅 설정

```python
import logging
from vanish.utils.logging import setup_logging

# 로그 레벨 설정
setup_logging(level=logging.INFO)

# 커스텀 로거
logger = logging.getLogger("vanish")
logger.setLevel(logging.DEBUG)
```

**로그 레벨:**
- `WARNING` (기본): 경고 및 에러만
- `INFO`: 진행 상황 포함
- `DEBUG`: 상세 디버그 정보

---

## 라이선스 및 참조

**라이선스:** MIT

**사용 모델:**
- Demucs v4 (Hybrid Transformer)
- Resemble-Enhance
- VoiceFixer

자세한 내용은 [README.md](../README.md) 참조
