# Audio Pre-Analysis Module

오디오 전처리 파이프라인의 첫 단계인 Pre-Analysis 모듈입니다.
입력 오디오의 특성을 분석하여 후속 처리 단계의 파라미터를 최적화합니다.

## 🎯 주요 기능

### 1. BGM 강도 측정
- 주파수 스펙트럼 분석으로 BGM vs 음성 비율 측정
- 결과에 따라 음원 분리 모델 비율 자동 조정

### 2. 잡음 유형 분류
- **정적 잡음** (Static): 에어컨, 팬 등 지속적 배경음
- **충격 잡음** (Impulse): 클릭, 팝 등 갑작스런 소리
- **하울링** (Howling): 주기적 공명음

### 3. 음성 대역폭 추정
- 음성 주파수 범위 측정
- 좁은 대역폭 감지 시 강화 처리 활성화

### 4. 볼륨 프로파일 분석
- 평균/피크 볼륨 측정
- 다이나믹 레인지 계산 및 압축 설정 자동화

## 📦 설치

```bash
# 의존성 설치
pip install -r requirements.txt
```

## 🚀 사용법

### Python API 사용

```python
from modules.pre_analyzer import PreAnalyzer
from config.analysis_config import AnalysisConfig

# 분석기 생성
config = AnalysisConfig(sample_rate=22050)
analyzer = PreAnalyzer(config=config)

# 분석 실행
result = analyzer.analyze('input.wav')

# 결과 출력
result.print_summary()

# 파이프라인 설정 생성
pipeline_config = result.to_pipeline_config()
print(pipeline_config)
```

### 커맨드라인 사용

```bash
# 기본 분석
python modules/pre_analyzer.py input.wav

# 샘플레이트 지정
python modules/pre_analyzer.py input.wav --sr 44100

# 파이프라인 설정 파일 저장
python modules/pre_analyzer.py input.wav --output-config config.json
```

## 📊 출력 예시

```
==================================================
Pre-Analysis Results
==================================================
├─ BGM Intensity: 45.2%
│  → Ensemble(MDX:65%, Demucs:35%)
├─ Noise Types:
│  ├─ Static: HIGH → DeepFilterNet(strength=0.8)
│  ├─ Impulse: LOW
│  └─ Howling: NONE
├─ Voice Bandwidth: 2.8kHz
│  → Enhancement(strength=1.0)
└─ Volume Profile:
   ├─ Avg: -18.5dB
   ├─ Peak: -3.2dB
   └─ Dynamic Range: 28.3dB
      → Compression(ratio=3:1)
==================================================
```

## 🔧 Pipeline Config 출력

```json
{
  "source_separation": {
    "mdx_ratio": 0.65,
    "demucs_ratio": 0.35,
    "reason": "Moderate BGM intensity"
  },
  "noise_reduction": {
    "strength": 0.8,
    "reason": "Static noise detected",
    "apply_gate": false
  },
  "voice_enhancement": {
    "strength": 1.0,
    "reason": "Narrow bandwidth (2800 Hz)"
  },
  "compression": {
    "ratio": 3.0,
    "threshold": -20,
    "attack": 5,
    "release": 50,
    "reason": "Large dynamic range (28.3 dB)"
  }
}
```

## 🧪 테스트

```bash
# 단위 테스트 실행
pytest tests/test_pre_analyzer.py -v

# 커버리지 포함
pytest tests/test_pre_analyzer.py --cov=modules --cov-report=html
```

## 📁 프로젝트 구조

```
analysis/
├── modules/
│   ├── __init__.py
│   └── pre_analyzer.py          # 메인 분석 모듈
├── config/
│   ├── __init__.py
│   └── analysis_config.py       # 설정 모델
├── utils/
│   ├── __init__.py
│   └── audio_utils.py           # 공통 유틸리티
├── tests/
│   └── test_pre_analyzer.py     # 단위 테스트
├── requirements.txt             # 의존성
└── README.md                    # 문서
```

## 🎓 분석 알고리즘

### BGM Intensity
```python
# 주파수 대역 에너지 계산
low_freq = 20-200Hz     # 베이스/드럼
mid_freq = 200-4000Hz   # 음성
high_freq = 4-20kHz     # 심벌/하이햇

# BGM 비율 = (저주파 + 고주파) / 전체
bgm_ratio = (low + high) / total
```

### Noise Classification
```python
# 정적 잡음: Spectral Flatness
flatness > 0.5 → static noise detected

# 충격 잡음: Zero-Crossing Rate spikes
zcr_variance > threshold → impulse noise

# 하울링: Sustained pitch detection
low_pitch_variance + high_voiced_prob → howling
```

### Voice Bandwidth
```python
# Spectral Rolloff (95% 에너지 포함 주파수)
bandwidth = percentile(rolloff, 95)

# 범위별 분류
< 3kHz  → 좁은 대역폭 (전화 품질)
3-5kHz  → 보통 대역폭
> 5kHz  → 넓은 대역폭 (HD 품질)
```

## ⚙️ 설정 커스터마이징

```python
from config.analysis_config import AnalysisConfig

config = AnalysisConfig(
    sample_rate=44100,        # 샘플레이트
    n_fft=4096,              # FFT 크기
    hop_length=1024,         # Hop 길이
    vad_top_db=30.0,         # VAD 임계값
    spike_threshold=3.5      # 스파이크 감지 임계값
)

analyzer = PreAnalyzer(config=config)
```

## 🔍 성능

- **분석 시간**: ~5초 (3분 오디오, CPU)
- **메모리 사용**: ~500MB (22.05kHz, 3분)
- **GPU**: 불필요 (CPU 분석으로 충분)

## 📝 라이선스

MIT License

## 👥 기여

이슈와 PR은 언제든지 환영합니다!
