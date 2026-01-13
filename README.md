# 39Agent

AI-Controlled Virtual Machine Agent. QEMU/KVM 기반 가상 컴퓨터를 AI가 완전히 조작할 수 있게 해주는 프로그램.

## 요구사항

- Python 3.11+
- QEMU/KVM
- espeak (TTS용)
- NVIDIA GPU (CUDA) 또는 AMD GPU (ROCm) - 권장

### espeak 설치 (TTS 필수)

```bash
# Ubuntu/Debian
sudo apt install espeak

# macOS
brew install espeak

# Fedora
sudo dnf install espeak
```

### QEMU 설치

```bash
# Ubuntu/Debian
sudo apt install qemu-system-x86 qemu-utils

# Fedora
sudo dnf install qemu-system-x86 qemu-img

# Arch Linux
sudo pacman -S qemu-full
```

## 실행

```bash
python run.py
```

첫 실행 시 자동으로:
1. 가상환경 생성 (`data/venv/`)
2. GPU 감지 (NVIDIA/AMD/CPU)
3. 적절한 PyTorch 버전 설치
4. 필요한 의존성 설치
5. Hugging Face에서 모델 다운로드

실행 후 `http://localhost:8039` 접속.

## 디렉토리 구조

```
39Agent/
├── run.py              # 실행 파일
├── requirements.txt    # Python 의존성
├── data/               # 데이터 (git 제외)
│   ├── venv/           # 가상환경
│   ├── iso/            # ISO 이미지
│   ├── llm_models/     # LLM 모델 (HuggingFace 캐시)
│   └── chromadb/       # 메모리 DB
├── character/          # 캐릭터 설정
│   └── default/
│       ├── config.yaml
│       ├── avatar.png  # 선택사항
│       ├── voice.wav   # 선택사항 (음성 클론용)
│       └── voice.txt   # voice.wav의 transcript (음성 클론 시 필수)
├── src/                # Python 소스
│   ├── server.py       # FastAPI 서버
│   ├── ai_agent.py     # AI 에이전트
│   ├── vm_manager.py   # VM 관리
│   ├── tools.py        # 도구 구현
│   ├── memory.py       # RAG 메모리
│   └── tts.py          # TTS 엔진
└── web/                # 웹 프론트엔드
    ├── static/
    └── templates/
```

## ISO 이미지 추가

`data/iso/` 폴더에 ISO 파일을 넣으면 웹 UI에서 선택 가능.

## 캐릭터 추가

`character/` 폴더에 새 폴더 생성:

```
character/mychar/
├── config.yaml
├── avatar.png   # 선택사항 - 캐릭터 이미지
├── voice.wav    # 선택사항 - 음성 클론용 (3-15초)
└── voice.txt    # voice.wav의 정확한 transcript (음성 클론 시 필수)
```

config.yaml 예시:
```yaml
display_name: My Character
llm_model: Qwen/Qwen3-VL-4B-Instruct
voice_file: voice.wav  # 같은 폴더 내 파일명 또는 절대경로
voice_text: voice.txt  # voice.wav의 transcript (생략 시 voice.txt 자동 탐색)

system_prompt: |
  You are a helpful assistant...

personality: |
  - Friendly and helpful
  - Always explains actions
```

## Vision-Language Models

Hugging Face 모델명을 입력하면 첫 실행 시 자동 다운로드됩니다.

| 모델 | 크기 | 설명 |
|------|------|------|
| `Qwen/Qwen3-VL-4B-Instruct` | ~8GB | 기본값, 균형잡힌 성능 |
| `unsloth/Qwen3-VL-4B-Instruct-GGUF` | ~4GB | 경량화 버전 |
| `Qwen/Qwen3-VL-4B-Instruct-GGUF` | ~4GB | 공식 GGUF |

## TTS (음성 합성)

[NeuTTS Air](https://huggingface.co/neuphonic/neutts-air-q8-gguf) 모델을 사용합니다. 0.5B LLM 기반의 on-device TTS로, CPU에서도 실시간 생성이 가능합니다.

### 기본 음성
`voice_file`을 설정하지 않으면 기본 음성으로 출력됩니다.

### 음성 클론 (Instant Voice Cloning)
3초 이상의 음성 파일과 해당 transcript를 제공하면 해당 음성을 클론합니다.

```yaml
voice_file: voice.wav  # 3-15초 분량의 깨끗한 음성 파일
voice_text: voice.txt  # voice.wav의 정확한 transcript
```

**voice.txt 예시:**
```
Hello, my name is Dave, and I'm from London.
```

### Reference Audio 가이드라인
- Mono channel
- 16-44 kHz sample rate
- 3-15초 길이
- .wav 형식
- 배경 소음 최소화
- 자연스럽고 연속적인 발화

## AI 도구

| 도구 | 설명 |
|------|------|
| `wait` | 지정 시간 대기 |
| `input` | 키 또는 키 조합 입력 (Enter, Ctrl+C 등) |
| `text` | 텍스트 타이핑 |
| `cursor-tp` | 커서를 절대 좌표로 이동 |
| `cursor-move` | 커서를 상대 좌표로 이동 |
| `click` | 마우스 클릭 |
| `click-on` | 마우스 버튼 누름 유지 |
| `click-off` | 마우스 버튼 해제 |
| `speak` | 사용자에게 말하기 (TTS) |
| `memory` | 이전 대화 검색 (RAG) |
| `todo` | 할 일 목록 관리 |
| `end` | 작업 종료 |

## VM 설정

기본 설정:
- 저장공간: 30GB
- RAM: 12GB
- VRAM: 125MB
- 해상도: 1280x720
- FPS: 30

웹 UI에서 조정 가능.

## 기술 스택

- **Backend**: FastAPI, WebSocket
- **VM**: QEMU/KVM, QMP (QEMU Monitor Protocol), VNC
- **AI**: Transformers (Qwen3-VL), ChromaDB (RAG)
- **TTS**: NeuTTS Air (neuphonic/neutts-air-q8-gguf) - On-device Voice Cloning
- **Frontend**: Vanilla JS, CSS
