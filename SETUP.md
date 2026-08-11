# Gemini Web Python Service — Setup, chạy và demo

Tài liệu này mô tả cách cài đặt service trên Windows bằng PowerShell, đăng nhập Gemini lần đầu, chạy service và kiểm tra end-to-end.

## 1. Yêu cầu

Khuyến nghị:

- Windows 10/11.
- Python 3.11+.
- Google Chrome đã cài.
- Tài khoản Google có thể sử dụng Gemini Web.
- Kết nối Internet.

Kiểm tra Python:

```powershell
py --version
```

hoặc:

```powershell
python --version
```

---

## 2. Giải nén project

Ví dụ:

```powershell
Expand-Archive .\gemini-web-python.zip .\gemini-web-python -Force
cd .\gemini-web-python
```

Project chính:

```text
gemini-web-python/
├── app.py
├── gemini_web.py
├── requirements.txt
├── .env.example
├── README.md
├── ARCHITECTURE.md
└── SETUP.md
```

---

## 3. Tạo virtual environment

```powershell
py -m venv .venv
```

Activate:

```powershell
.\.venv\Scripts\Activate.ps1
```

Nếu PowerShell chặn script activation, có thể chạy cho session hiện tại:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

Kiểm tra:

```powershell
python --version
pip --version
```

---

## 4. Cài dependency

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Cài Playwright Chromium để có browser fallback:

```powershell
playwright install chromium
```

Service mặc định thử Google Chrome trước:

```env
BROWSER_CHANNEL=chrome
```

Nếu Chrome channel không chạy được, code sẽ fallback sang Chromium do Playwright quản lý.

---

## 5. Cấu hình environment

Có thể copy file mẫu:

```powershell
Copy-Item .env.example .env
```

File mẫu:

```env
HOST=127.0.0.1
PORT=8787
GEMINI_PROFILE_DIR=var/gemini-profile
GEMINI_TIMEOUT_MS=120000
HEADLESS=false
BROWSER_CHANNEL=chrome
```

### Lưu ý quan trọng

Bản hiện tại đọc biến môi trường bằng `os.getenv()`, không tự load `.env` bằng `python-dotenv`.

Nếu muốn thay giá trị khi chạy PowerShell, set biến trực tiếp trước khi start:

```powershell
$env:PORT="8787"
$env:HEADLESS="false"
$env:BROWSER_CHANNEL="chrome"
```

Nếu giữ giá trị mặc định thì không cần set gì thêm.

---

## 6. Bootstrap đăng nhập Gemini lần đầu

Chạy:

```powershell
python app.py bootstrap
```

Service sẽ:

1. Tạo hoặc mở profile tại `var/gemini-profile`.
2. Mở Chrome/Chromium.
3. Điều hướng tới `https://gemini.google.com/app`.
4. Chờ bạn hoàn tất login.

Trong browser:

- Đăng nhập Google nếu chưa đăng nhập.
- Xử lý 2FA/security prompt nếu có.
- Đảm bảo đã vào được giao diện Gemini và có thể thấy ô nhập prompt.

Sau đó quay lại terminal và nhấn:

```text
ENTER
```

Nếu thành công, terminal sẽ báo profile đã được lưu và browser đóng lại.

### Kiểm tra profile đã tạo

```powershell
Get-ChildItem .\var\gemini-profile
```

Không xóa thư mục này nếu muốn giữ trạng thái đăng nhập.

---

## 7. Chạy service

```powershell
python app.py serve
```

Mặc định:

```text
http://127.0.0.1:8787
```

Giữ terminal này chạy trong lúc demo.

---

## 8. Demo bằng giao diện web

Mở browser:

```text
http://127.0.0.1:8787/
```

UI gồm:

- Model label.
- Nút `Check health`.
- Ô nhập prompt.
- Nút `Send`.
- Vùng hiển thị response.

### Test 1 — Health

Bấm:

```text
Check health
```

Kỳ vọng khi session còn hợp lệ:

```json
{
  "ok": true,
  "auth": "ok",
  "url": "https://gemini.google.com/..."
}
```

### Test 2 — Prompt

Nhập:

```text
Chỉ trả lời chính xác: GEMINI_WEB_OK
```

Bấm `Send`.

Nếu browser automation và DOM selector đang hoạt động đúng, UI phải trả response Gemini về vùng kết quả.

---

## 9. Demo API bằng PowerShell

### Health

```powershell
Invoke-RestMethod http://127.0.0.1:8787/health
```

Hoặc:

```powershell
curl.exe http://127.0.0.1:8787/health
```

---

## 10. Demo `/api/generate`

```powershell
$body = @{
    model  = "gemini-web"
    prompt = "Chỉ trả lời chính xác: GEMINI_WEB_SERVICE_OK"
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://127.0.0.1:8787/api/generate" `
    -Method POST `
    -ContentType "application/json" `
    -Body $body
```

Response dự kiến về shape:

```json
{
  "model": "gemini-web",
  "text": "GEMINI_WEB_SERVICE_OK"
}
```

---

## 11. Demo OpenAI-compatible endpoint

Request:

```powershell
$body = @{
    model = "gemini-web"
    messages = @(
        @{
            role = "system"
            content = "Trả lời cực ngắn."
        },
        @{
            role = "user"
            content = "Chỉ trả lời chính xác: OPENAI_COMPAT_OK"
        }
    )
    stream = $false
} | ConvertTo-Json -Depth 10

Invoke-RestMethod `
    -Uri "http://127.0.0.1:8787/v1/chat/completions" `
    -Method POST `
    -ContentType "application/json" `
    -Body $body
```

Response có dạng:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 0,
  "model": "gemini-web",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "OPENAI_COMPAT_OK"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": null
}
```

`stream=true` hiện chưa được implement.

---

## 12. Test quan trọng nhất — Session persistence

Mục tiêu: chứng minh login chỉ cần một lần khi Google chưa revoke session.

### Bước 1

Bootstrap và call Gemini thành công ít nhất một lần.

### Bước 2

Dừng server:

```text
Ctrl + C
```

### Bước 3

Không chạy lại bootstrap.

Khởi động lại:

```powershell
python app.py serve
```

### Bước 4

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8787/health
```

### Bước 5

Call Gemini lại:

```powershell
$body = @{
    model  = "gemini-web"
    prompt = "Chỉ trả lời chính xác: LOGIN_PERSISTED"
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://127.0.0.1:8787/api/generate" `
    -Method POST `
    -ContentType "application/json" `
    -Body $body
```

Nếu request thành công mà không login lại, persistence đã hoạt động đúng.

---

## 13. Chạy với browser hiện ra hay headless

### Khi setup/debug

Khuyến nghị:

```powershell
$env:HEADLESS="false"
python app.py serve
```

Ưu điểm: nhìn trực tiếp Gemini Web và dễ biết selector/session lỗi ở đâu.

### Khi đã ổn định

Có thể thử:

```powershell
$env:HEADLESS="true"
python app.py serve
```

Nếu Google/Gemini có hành vi khác khi headless, quay lại `HEADLESS=false`.

---

## 14. Đổi profile directory

Ví dụ tạo profile khác:

```powershell
$env:GEMINI_PROFILE_DIR="var/gemini-profile-account-2"
python app.py bootstrap
```

Sau này khi chạy service với account đó phải dùng cùng biến:

```powershell
$env:GEMINI_PROFILE_DIR="var/gemini-profile-account-2"
python app.py serve
```

---

## 15. Lỗi thường gặp

### `AUTH_REQUIRED`

Biểu hiện:

- HTTP 401.
- Health trả `auth=required`.

Cách xử lý:

```powershell
python app.py bootstrap
```

---

### Chrome không mở

Thử đảm bảo Chromium của Playwright đã được cài:

```powershell
playwright install chromium
```

Sau đó có thể ép fallback bằng cách bỏ Chrome channel cho process hiện tại:

```powershell
$env:BROWSER_CHANNEL=""
python app.py bootstrap
```

---

### Profile đang bị lock

Không chạy cùng lúc:

```text
python app.py bootstrap
```

và:

```text
python app.py serve
```

trên cùng `GEMINI_PROFILE_DIR`.

Đóng Chrome/service đang giữ profile rồi thử lại.

---

### `Could not find Gemini prompt box`

Gemini Web có thể đã thay DOM hoặc đang hiển thị một màn hình trung gian.

Test với:

```powershell
$env:HEADLESS="false"
python app.py serve
```

Quan sát browser.

Nếu giao diện Gemini đã đổi, kiểm tra selector trong:

```python
GeminiWebClient._find_prompt_box()
```

file:

```text
gemini_web.py
```

---

### Có prompt gửi đi nhưng không lấy được response

Kiểm tra:

```python
GeminiWebClient._all_response_texts()
```

Gemini có thể đã thay cấu trúc response DOM.

---

### Timeout khi model trả lời lâu

Tăng timeout:

```powershell
$env:GEMINI_TIMEOUT_MS="240000"
python app.py serve
```

---

### Port 8787 đã được dùng

Đổi port:

```powershell
$env:PORT="8790"
python app.py serve
```

Sau đó mở:

```text
http://127.0.0.1:8790
```

---

## 16. Checklist demo hoàn chỉnh

```text
[ ] Python chạy được
[ ] pip install -r requirements.txt thành công
[ ] playwright install chromium thành công
[ ] python app.py bootstrap mở browser
[ ] Gemini login thành công
[ ] var/gemini-profile được tạo
[ ] python app.py serve start thành công
[ ] GET /health => auth=ok
[ ] UI gửi prompt thành công
[ ] POST /api/generate thành công
[ ] POST /v1/chat/completions thành công
[ ] Restart service không cần bootstrap lại
```

---

## 17. Cách reset sạch session

Chỉ làm khi muốn bỏ toàn bộ login state của profile hiện tại.

Đảm bảo service/browser đã dừng, sau đó:

```powershell
Remove-Item -Recurse -Force .\var\gemini-profile
```

Bootstrap lại:

```powershell
python app.py bootstrap
```

---

## 18. Chạy lại sau khi sửa code

Sau mỗi lần sửa `app.py` hoặc `gemini_web.py`:

1. Dừng process đang chạy bằng `Ctrl+C`.
2. Start lại:

```powershell
python app.py serve
```

3. Gọi `/health`.
4. Gửi một prompt test thật.

Không nên chỉ nhìn source rồi kết luận runtime đã load code mới.
