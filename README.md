# Hot Topic Finder

웹페이지를 읽고 LLM을 위한 요약 llms.txt 파일을 자동으로 생성하는 에이전트 기반 도구입니다.

## 개요

Hot Topic Finder는 LangGraph를 활용한 에이전트 기반 시스템으로, 웹사이트 URL을 입력하면 해당 웹사이트를 분석하여 LLM(대규모 언어 모델)이 필요로 할 때 참조할 수 있는 llms.txt 파일을 생성합니다. 이 도구는 다음과 같은 특징을 가지고 있습니다:

- 웹페이지를 지능적으로 탐색하고 분석
- 관련 링크를 따라가며 웹사이트의 다양한 부분을 탐색
- 각 페이지의 핵심 내용을 추출하고 분석
- LLM을 위한 구조화된 llms.txt 파일 생성

## 설치 방법

### 요구사항

- Python 3.8 이상
- OpenAI API 키

### 설치 단계

1. 저장소를 클론하거나 다운로드합니다:

```bash
git clone https://github.com/yourusername/hot_topic_finder.git
cd hot_topic_finder
```

2. 필요한 패키지를 설치합니다:

```bash
pip install -r requirements.txt
```

3. OpenAI API 키를 설정합니다. `.env` 파일을 생성하고 API 키를 추가합니다:

```
OPENAI_API_KEY=your_api_key_here
```

## 사용 방법

### 명령줄 인터페이스

```bash
python hot_topic_finder.py https://example.com -o output.txt
```

옵션:
- `url`: 탐색할 웹사이트 URL (필수)
- `-o, --output`: 결과를 저장할 파일 경로 (기본값: llms.txt)
- `-v, --verbose`: 상세 로그 출력 활성화

### 파이썬 코드에서 사용

```python
from src.main import generate_llms_txt

# llms.txt 생성
result = generate_llms_txt("https://example.com", "output.txt")

# 결과 출력
print(result)
```

## 결과 예시

생성된 llms.txt 파일은 다음과 같은 형식을 가집니다:

```
# Example.com
https://example.com/page1: LLM should read this page when needing information about topic X, especially details on A and B.
https://example.com/page2: LLM should read this page when looking for information about Y, including its relation to Z.
```

## 작동 방식

Hot Topic Finder는 LangGraph를 사용한 상태 기반 에이전트 시스템으로 다음 단계로 작동합니다:

1. **초기화**: 입력된 URL을 검증하고 탐색 상태를 초기화합니다.
2. **웹페이지 탐색**: 웹페이지를 크롤링하고 콘텐츠와, 관련 링크를 추출합니다.
3. **페이지 분석**: 추출된 콘텐츠를 분석하여 핵심 주제와 중요 포인트를 파악합니다.
4. **llms.txt 생성**: 분석 결과를 기반으로 LLM을 위한 구조화된 llms.txt 파일을 생성합니다.

이 과정은 모두 자동화되어 있으며, 각 단계는 LangGraph의 노드와 엣지를 통해 조율됩니다.

## 제한 사항

- 현재 버전은 기본적인 HTML 웹페이지만 지원합니다.
- JavaScript로 렌더링되는 콘텐츠는 완전히 캡처되지 않을 수 있습니다.
- 속도와 리소스 제한으로 인해 기본적으로 최대 10개의 페이지만 탐색합니다.

## 라이선스

MIT 라이선스 