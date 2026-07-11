# RAG Pipeline Review

Ngay cap nhat: 2026-07-11

Tai lieu nay mo ta pipeline RAG hien tai cua IELTS chatbot, nhung huong da va dang trien khai, cung cac van de con ton tai de review tiep. Muc tieu cua pipeline khong phai toi uu cho rieng mot file PDF mau, ma la tao nen mot luong xu ly tong quat cho phan lon tai lieu IELTS Reading/Text/PDF/DOCX/anh.

## 1. Muc tieu hien tai cua RAG

Chatbot hien tai co hai che do tra loi chinh:

1. Tra loi truc tiep bang LLM khi cau hoi khong can tai lieu.
2. Dung RAG khi cau hoi co lien quan den tai lieu da upload.

Trong huong moi, RAG khong chi la vector search tren text phang. Pipeline can hieu duoc cau truc tai lieu hoc IELTS:

- Tai lieu gom nhung passage nao.
- Moi passage nam o trang nao.
- Moi passage co nhom cau hoi nao.
- Nhom cau hoi co dang bai nao.
- Cau hoi cu the thuoc nhom nao va passage nao.
- Khi user chi yeu cau xem/dich/giai thich cau hoi thi khong duoc tu giai bai.
- Khi user yeu cau tra loi/giai bai thi moi dua passage evidence vao LLM.

Muc tieu dai han:

```text
Tim passage va so cau bang cau truc.
Tim bang chung bang semantic retrieval.
Chi de LLM giai bai khi user that su yeu cau.
```

## 2. Kien truc tong quan hien tai

Luong tong the hien tai:

```text
User upload file
    |
    v
POST /documents/upload
    |
    v
Luu file tam trong upload dir
    |
    v
DocumentProcessor.process_file(...)
    |
    +--> FileRouter
    |       +--> text / markdown
    |       +--> pdf
    |       +--> docx
    |       +--> image
    |
    +--> Extractor theo loai file
    |       +--> native text
    |       +--> OCR khi can
    |
    +--> NativeOCRReconciler
    |       +--> gop native va OCR
    |       +--> bo duplicate
    |       +--> giu alternative_sources de debug
    |
    +--> IELTSStructureParser
    |       +--> passage
    |       +--> question_group
    |       +--> question
    |       +--> document_outline
    |
    +--> StructuredChunker
            +--> document_outline chunk
            +--> passage chunk
            +--> question_group chunk
            +--> question chunk
    |
    v
LocalVectorStore.upsert(...)
    |
    +--> data/rag/documents.json
    +--> data/rag/embeddings.npy
    |
    v
User chat
    |
    v
POST /chat hoac POST /chat/stream
    |
    v
store.probe_with_catalog(...)
    |
    +--> document catalog
    +--> overview probe
    +--> dense search
    +--> keyword search
    +--> question range search
    |
    v
detect_query_intent(...)
    |
    +--> document_overview
    +--> show_questions
    +--> translate_questions
    +--> explain_questions
    +--> solve_questions
    +--> semantic_qa
    +--> direct
    |
    v
select sources + optional passage expansion
    |
    v
rag_prompt(...)
    |
    v
Qwen3-4B-Instruct-2507 qua Ollama
    |
    v
Frontend render + debug pipeline
```

## 3. File va component lien quan

### Backend API

- `backend/app/main.py`
  - Tao FastAPI app.
  - Entry point upload tai lieu.
  - Entry point chat thuong va chat streaming.
  - Goi `DocumentProcessor`.
  - Goi `LocalVectorStore`.
  - Goi route classifier, intent detector va prompt builder.
  - Tao debug response cho frontend.

### Document pipeline

- `backend/app/document_pipeline/processor.py`
  - Dieu phoi toan bo qua trinh xu ly file.
  - Khoi tao router, OCR, reconciler, IELTS parser, structured chunker, fallback semantic chunker.

- `backend/app/document_pipeline/models.py`
  - Schema chinh:
    - `PageQuality`
    - `DocumentElement`
    - `ProcessedPage`
    - `ProcessedDocument`
    - `DocumentChunk`

- `backend/app/document_pipeline/routing.py`
  - Xac dinh file route: text/pdf/docx/image.
  - Hien tai chi can ho tro Text, Markdown, PDF, DOCX va anh.

- `backend/app/document_pipeline/extractors/pdf.py`
  - Trich xuat PDF.
  - Uu tien native text.
  - Goi OCR khi native text yeu, scan, hoac can bo sung noi dung.

- `backend/app/document_pipeline/extractors/docx.py`
  - Doc DOCX truc tiep.
  - Lay heading/paragraph/table va cac thanh phan co the trich xuat bang `python-docx`.

- `backend/app/document_pipeline/extractors/image.py`
  - Xu ly anh upload truc tiep.
  - Goi OCR de tao element text.

- `backend/app/document_pipeline/extractors/text.py`
  - Xu ly text/markdown.

- `backend/app/document_pipeline/ocr.py`
  - OCR processor.
  - Uu tien PaddleOCR.
  - Co fallback medium va fallback Tesseract neu duoc cau hinh.
  - Co warmup de model lon san sang truoc request that.

- `backend/app/document_pipeline/reconciliation.py`
  - Giai quyet trung lap native text va OCR overlay.
  - Muc tieu: native tot thi native la canonical; OCR chi bo sung vung thieu hoac thay the khi native hong.

- `backend/app/document_pipeline/ielts.py`
  - Parser cau truc IELTS.
  - Chuyen `ProcessedDocument` thanh `IELTSDocument`.
  - Tao cac object passage, question group, question.
  - Tao structured chunks.

- `backend/app/document_pipeline/chunking.py`
  - Fallback semantic chunker cu.
  - Duoc dung khi structured parser tat hoac structured chunker khong tao duoc chunk.

### RAG store va retrieval

- `backend/app/rag.py`
  - `LocalVectorStore`.
  - Load/save `data/rag/documents.json`.
  - Load/save `data/rag/embeddings.npy`.
  - Embed bang `BAAI/bge-m3`.
  - Search bang dense vector.
  - Probe ket hop dense/keyword/question/overview.
  - Tao document catalog cho router va debug.
  - Overview retrieval.
  - Passage expansion cho solve intent.

### Intent va prompt

- `backend/app/intent.py`
  - Parse question range.
  - Detect query intent.
  - Filter sources theo intent.
  - Dedupe sources.

- `backend/app/llm.py`
  - Prompt direct.
  - Prompt RAG.
  - Router prompt.
  - Goi Ollama.
  - Streaming Ollama.
  - Prompt echo guard.

### Frontend

- `frontend/src/App.jsx`
  - UI chat.
  - Upload file.
  - Stream response.
  - Render markdown.
  - Debug pipeline.
  - Download debug tam thoi: cau hoi, cau tra loi, debug, sources, source previews.

- `frontend/src/styles.css`
  - Style chat va debug panel.

## 4. Luong upload va ingestion chi tiet

### 4.1 Upload endpoint

Endpoint chinh:

```text
POST /documents/upload
```

Tren backend:

1. Nhan `UploadFile`.
2. Luu file vao thu muc upload tam.
3. Goi:

```python
DOCUMENT_PROCESSOR.process_file(...)
```

4. Nhan ve:

```python
ProcessedDocument, list[DocumentChunk]
```

5. Ghi chunks vao vector store:

```python
get_store().upsert(chunks, source_file=filename)
```

6. Tra ve metadata/debug cho frontend:
   - document id
   - filename
   - so page
   - so chunk
   - route xu ly
   - extraction/structure report neu co

### 4.2 File routing

`FileRouter` xac dinh route theo loai file:

- `text`
- `pdf`
- `docx`
- `image`

Huong hien tai khong mo rong sang cac loai khac. Neu gap `.doc` legacy thi huong mong muon la convert sang DOCX/PDF bang LibreOffice headless, nhung day chua phai trong tam hien tai.

### 4.3 PDF extraction

Pipeline PDF muc tieu:

```text
PDF page
    |
    +--> Native text extraction
    |
    +--> Page quality evaluation
    |
    +--> Neu native tot:
    |       dung native lam canonical
    |
    +--> Neu native thieu/scan/chat luong kem:
            render page/region
            OCR bang PaddleOCR
            fallback neu can
```

Nguyen tac:

- Khong OCR toan bo PDF mac dinh.
- Khong chap nhan native text chi vi text khong rong.
- OCR chi nen bo sung cac page/region can thiet.
- Neu native va OCR trung noi dung, khong embed ca hai.
- Neu OCR chi la alternative source, giu de debug nhung khong chunk rieng.

Trang PDF duoc chuyen thanh:

```python
ProcessedPage(
    page_number=...,
    processing_route=...,
    quality_score=...,
    elements=[...],
    metadata={...},
)
```

Element duoc chuyen thanh:

```python
DocumentElement(
    element_id=...,
    page=...,
    type=...,
    raw_text=...,
    normalized_text=...,
    source=...,
    confidence=...,
    bbox=...,
    metadata={...},
)
```

### 4.4 OCR

Huong uu tien cua OCR:

```text
PaddleOCR small
    |
    +--> neu chat luong dat: dung ket qua
    |
    +--> neu chat luong thap: PaddleOCR medium
    |
    +--> neu van fail va config cho phep: Tesseract fallback
```

Luu y quan trong:

- Tesseract khong phai huong chinh.
- PaddleOCR la huong uu tien.
- Model OCR can warmup truoc khi request that neu chay Colab hoac moi truong GPU/CPU yeu.
- Khong nen bat tat ca model nang mac dinh.
- Table/layout/structure recognition chi nen chay khi page/region that su can.

### 4.5 DOCX extraction

DOCX duoc doc truc tiep, khong convert thanh anh.

Can giu duoc:

- heading
- paragraph
- list
- table
- hyperlink neu co
- header/footer neu can
- anh nhung neu co

Anh nhung trong DOCX nen duoc phan loai:

- logo/icon/trang tri: bo qua OCR
- screenshot co chu: OCR
- scan tai lieu: OCR/layout
- bieu do: OCR label + metadata
- anh khong co chu: giu caption/metadata neu co

Muc nay hien moi o muc co ban, chua phai pipeline Document AI hoan chinh.

### 4.6 Text/Markdown extraction

Text va Markdown duoc xu ly nhu plain text co cau truc nhe. Markdown nen giu heading/list/table neu co the, nhung hien tai chua co parser Markdown semantic manh nhu Docling.

## 5. Native-OCR reconciliation

Van de cu:

```text
Native text paragraph
OCR overlay paragraph
```

co the cung ton tai nhu hai element doc lap. Khi chunk/embed rieng, retrieval co the lay ca hai ket qua trung nhau, lam:

- ton context
- gay nhieu cho LLM
- lam LLM tuong noi dung lap lai la quan trong
- sai retrieval ranking

Huong da trien khai:

```text
Native + OCR elements
    |
    v
NativeOCRReconciler
    |
    +--> so cung page
    +--> so bbox neu co
    +--> so text similarity/token overlap
    |
    v
Canonical elements
    |
    +--> duplicate OCR duoc dua vao alternative_sources
    +--> khong chunk/embed duplicate
```

Ket qua mong muon:

```json
{
  "type": "paragraph",
  "source": "native_pdf",
  "confidence": 0.98,
  "normalized_text": "canonical text",
  "metadata": {
    "alternative_sources": [
      {
        "source": "paddleocr_small",
        "confidence": 0.91,
        "text": "..."
      }
    ]
  }
}
```

Dieu can luu y:

- Reconciliation phai tong quat, khong dua vao noi dung file cu the.
- Neu native va OCR khac nhau that su, khong duoc xoa tuy tien.
- Neu OCR bo sung table/image region, can giu no nhu noi dung bo sung.
- Neu bbox khong co, text similarity phai than trong hon.

## 6. IELTS structure parser hien tai

### 6.1 Vi sao can parser cau truc?

Neu chi dung text phang, chunking se de bi loi:

- Passage va questions bi tron.
- Instruction bi tach khoi question.
- Question group bi cat ngang.
- Overview chi lay passage similarity cao nhat.
- Query "Questions 1-4" phai dua vao vector search thay vi exact structure.
- LLM de tu giai khi user chi muon xem noi dung cau hoi.

Vi vay them buoc:

```text
ProcessedDocument
    |
    v
IELTSStructureParser
    |
    v
IELTSDocument
```

### 6.2 Schema logic

Schema trong `ielts.py` hien huong toi:

```text
IELTSDocument
    |
    +--> IELTSPassage
            |
            +--> title
            +--> paragraphs
            +--> question_groups
                    |
                    +--> IELTSQuestionGroup
                            |
                            +--> question_start
                            +--> question_end
                            +--> instructions
                            +--> question_type
                            +--> questions
                                    |
                                    +--> IELTSQuestion
```

Metadata quan trong:

- `passage_number`
- `passage_title`
- `question_range`
- `question_type`
- `page_numbers`
- `source_element_ids`
- `parent_id`

### 6.3 Cach parser nhan dien cau truc

Parser hien tai dung heuristic tong quat, khong duoc hard-code theo file cu the.

Cac tin hieu co the dung:

- Cum `Reading Passage`, `Passage 1`, `Passage 2`, ...
- Cum `Questions 1-4`, `Questions 18-23`, ...
- Sequence so cau hoi.
- Instruction IELTS:
  - True/False/Not Given
  - Yes/No/Not Given
  - Choose the correct letter
  - Complete the table
  - Complete the flow-chart
  - Answer the questions
  - Match each heading
- Thu tu page va element.
- Line boundary.
- Title-like text.
- Do dai dong.
- Body-like continuation.
- Loai tru footer/header/page marker.

Nguyen tac chong overfit:

- Khong hard-code `Make That Wine!`, `Destination Mars`, `Australia`, `Mars`, hoac bat ky topic nao.
- Khong hard-code file name.
- Khong hard-code rang passage 1 luon la cau 1-13.
- Khong hard-code answer key.
- Khong hard-code trang 1/2/3 cho passage nao.
- Khong hard-code pattern theo mot PDF mau.

Nhung diem parser co the tam dung:

- Pattern cau truc IELTS pho bien.
- Rule theo hinh thuc tai lieu, khong theo noi dung topic.
- Scoring title/passages dua tren tin hieu tong quat.
- Validator phat hien missing/duplicate/unassigned question.

### 6.4 Ket qua parser mong muon

Voi tai lieu IELTS Reading, parser can tao duoc outline tuong tu:

```text
Document outline
    Passage 1: <title>
        Questions 1-4: true_false_not_given
        Questions 5-10: table_completion
        Questions 11-13: multiple_choice

    Passage 2: <title>
        Questions 14-17: short_answer
        Questions 18-23: flowchart_completion
        Questions 24-26: multiple_choice

    Passage 3: <title>
        Questions 27-30: true_false_not_given
        Questions 31-35: table_completion
        Questions 36-40: short_answer
```

Day chi la vi du cau truc. He thong khong duoc mac dinh moi tai lieu deu co dung cac range nay.

## 7. Structured chunking hien tai

Sau parser, `StructuredChunker` tao chunk theo unit semantic thay vi cat theo page/token don thuan.

### 7.1 Cac loai chunk

Hien co cac loai chinh:

- `document_outline`
- `passage`
- `question_group`
- `question`

Moi chunk co:

```python
DocumentChunk(
    chunk_id=...,
    document_id=...,
    source_file=...,
    pages=[...],
    element_ids=[...],
    heading_path=[...],
    text=...,
    retrieval_text=...,
    display_text=...,
    token_count=...,
    min_confidence=...,
    metadata={...},
)
```

### 7.2 `retrieval_text` va `display_text`

Can tach hai loai text:

- `retrieval_text`: text co them metadata de embedding/search tot hon.
- `display_text`: text gan voi noi dung goc hon, de dua vao prompt va hien cho user.

Vi du:

```text
retrieval_text:
IELTS Academic Reading.
Passage 1.
Question Group 1-4.
Question Type: true_false_not_given.
1. ...
2. ...

display_text:
Questions 1-4
Do the following statements agree ...
1. ...
2. ...
```

### 7.3 Loi chunking can tranh

- Chunk chi chua footer/page number.
- Chunk chi chua title.
- Chunk passage bi dinh cau hoi cua passage khac.
- Chunk question group bi thieu instruction.
- Chunk individual question bi mat so cau.
- Chunk dai qua lon, lam loang context.
- Chunk native va OCR duplicate cung duoc embed.

## 8. Vector store va index

Store hien tai la local store:

```text
data/rag/documents.json
data/rag/embeddings.npy
```

`documents.json` chua list chunk da serialize:

- `chunk_id`
- `document_id`
- `source_file`
- `pages`
- `element_ids`
- `heading_path`
- `text`
- `retrieval_text`
- `display_text`
- `token_count`
- `min_confidence`
- `chunk_index`
- `metadata`

`embeddings.npy` chua vector tu `retrieval_text` neu co, nguoc lai dung `text`.

Embedding model:

```text
BAAI/bge-m3
```

LLM:

```text
Qwen3-4B-Instruct-2507 qua Ollama
```

OCR:

```text
PaddleOCR la huong chinh
Tesseract chi la fallback tuy cau hinh
```

## 9. Retrieval hien tai

### 9.1 Probe truoc khi route

Khi user chat, backend goi:

```python
store.probe_with_catalog(message, top_k)
```

Probe tra ve:

- `results`
- `has_hits`
- `has_strong_hits`
- `has_document_intent`
- `is_overview`
- `top_score`
- `top_keyword_score`
- `top_question_score`
- `top_overview_score`

Catalog tra ve:

- source file
- so chunks
- pages
- document ids
- mime types
- unit types
- passage numbers

Muc tieu cua catalog la giup router biet dang co tai lieu nao, co chunk nao, co passage/question metadata nao, thay vi quyet dinh direct/RAG trong chan khong.

### 9.2 Dense search

Dense search dung embedding BGE-M3.

Manh o:

- cau hoi semantic
- hoi noi dung theo chu de
- tim bang chung lien quan trong passage

Yeu o:

- query theo so cau cu the
- overview toan tai lieu
- cau hoi can exact lookup
- khi chunking sai hoac passage/question bi tron

### 9.3 Keyword search

Keyword search bo sung cho dense search.

Manh o:

- keyword xuat hien truc tiep
- ten passage
- cum tu trong cau hoi

Yeu o:

- score hien tai con tho.
- co the hien `keyword 2.0` ma khong noi ro do dau.
- khong nen dung keyword nhu co che chinh de quyet dinh cau truc.

### 9.4 Question range search

Question range search dung metadata:

```text
question_range=[start, end]
unit_type=question_group/question
```

Muc tieu:

- Query `Questions 1-4` phai lay dung question group 1-4.
- Query `cau 20` phai lay dung individual question 20 hoac group chua cau 20.
- Khong phu thuoc vector similarity.

Day la huong dung, nhung can tiep tuc nang cap thanh structured index ro rang hon.

### 9.5 Overview retrieval

Voi query tong quan nhu:

- `Noi dung tai lieu la gi?`
- `Tai lieu tren gom nhung passage nao?`
- `Tom tat file nay`

He thong khong nen lay top-k dense binh thuong, vi model co the chi nhin mot passage co similarity cao.

Huong hien tai:

```text
overview query
    |
    v
store.overview(...)
    |
    +--> document_outline
    +--> chunk passage dau tien cua moi passage
```

Can tiep tuc mo rong:

- Query `tai lieu gom nhung passage nao` nen route chac chan vao `document_overview` hoac structured overview.
- Overview nen uu tien outline + summary/passage representative, khong dua qua dense ranking.

### 9.6 Parent/passage expansion

Khi intent la `solve_questions`, source ban dau co the la question chunk. He thong can bo sung passage context de LLM co evidence.

Luong hien tai:

```text
Question source
    |
    v
passage_context_for_sources(...)
    |
    v
Add passage chunks cung passage_number
```

Muc tieu:

- Show/explain question: chi can question context.
- Solve question: can question + passage evidence.

## 10. Routing va intent policy

### 10.1 Route direct/RAG

Hien tai route co hai tang:

1. Deterministic/probe-based:
   - Neu probe cho thay document intent, route RAG.
   - Neu overview/question score manh, route RAG.

2. LLM router:
   - Neu khong ro, goi `classify_route(...)`.
   - Router duoc cung cap document catalog + retrieval probe.

Day la cai tien so voi pattern router cu, vi router biet:

- Dang co file nao.
- Co bao nhieu chunk.
- Co passage/question metadata khong.
- Retrieval probe co hit khong.

### 10.2 Query intent hien tai

`detect_query_intent(...)` hien chia:

- `document_overview`
- `show_questions`
- `translate_questions`
- `explain_questions`
- `solve_questions`
- `semantic_qa`
- `direct`

Y nghia:

```text
show_questions
    User muon xem noi dung cau hoi.
    Khong duoc giai.

translate_questions
    User muon dich cau hoi.
    Khong duoc giai.

explain_questions
    User muon giai thich de bai/dang bai/tu vung.
    Khong duoc dua dap an.

solve_questions
    User muon tra loi/giai bai.
    Moi duoc tim evidence va dua dap an.

semantic_qa
    User hoi noi dung tai lieu theo y nghia.

document_overview
    User hoi tong quan tai lieu.
```

### 10.3 Diem yeu cua intent hien tai

Intent hien tai van con dua nhieu vao pattern:

- `tra loi`
- `dap an`
- `giai cau`
- `dịch`
- `question 1-4`
- `cau 1 den 4`

Day la tam chap nhan de tach policy nhanh, nhung khong du manh ve lau dai.

Can tranh:

- Them rule qua cu the cho mot file.
- Them pattern theo topic.
- Them pattern theo vi du user da hoi.
- Dung keyword duy nhat de quyet dinh hanh vi quan trong.

Huong dung:

- Deterministic parser cho question range.
- Intent classifier co confidence.
- Rule dua tren action cua user, khong dua tren noi dung file.
- Neu ambiguous, default an toan:
  - neu user hoi `noi dung cau hoi` thi show, khong solve.
  - neu user hoi `tra loi/giai/answer` thi solve.
  - neu context khong co passage evidence thi noi khong du evidence.

## 11. Prompt va generation policy

Prompt RAG hien gom:

- Assistant style cho IELTS Vietnamese learner.
- Bat buoc dung study material context.
- Khong invent passage/question/person/date/example.
- Neu khong co context, noi khong tim thay trong tai lieu.
- Cau hoi statements chi la prompt, khong phai evidence.
- Cite source file va page marker.
- Generation policy theo `query_intent`.

### 11.1 Show questions

Policy:

- Chi liet ke instruction va statement.
- Khong danh gia statement.
- Khong dua True/False/Not Given.
- Khong dung passage evidence.
- Co the them nghia tieng Viet ngan gon.

Van de da gap:

- Model van co luc lay question statement lam evidence.
- Model van co luc giai luon khi user chi hoi `noi dung Questions 1-4`.

Nguyen nhan:

- Context co ca question text va passage text.
- Conversation history co the gay nhieu.
- Prompt dang duoc gui qua completion style `/api/generate`.
- Model co xu huong "tutor" nen tu giai.

Huong da lam:

- Tach intent.
- Voi show/translate, bo history khoi prompt.
- Them policy cam giai ro hon.
- Filter sources de uu tien question chunks.

Can tiep tuc:

- Co deterministic renderer cho show-only neu can chat luong tuyet doi.
- Hoac van dung LLM de lam muot, nhung context chi chua question group/question, khong chua passage evidence.
- Them output validator: neu show_questions ma output chua `TRUE/FALSE/NOT GIVEN` nhu dap an thi reject/regenerate.

### 11.2 Solve questions

Policy:

- User phai that su yeu cau tra loi/giai bai.
- Phai dua passage evidence.
- Voi True/False/Not Given:
  - so statement voi passage evidence
  - neu passage ung ho: TRUE
  - neu passage mau thuan: FALSE
  - neu khong co thong tin: NOT GIVEN
- Neu context chi co question text ma khong co passage evidence, phai bao khong du evidence.

Van de hien tai:

- Giai thich co luc dung nhung chua du ro.
- Can trich evidence ngan hon, ro hon.
- Can phan biet "khong co thong tin" va "sai" ky hon.
- Can tranh hallucinate quote khong co trong context.

### 11.3 Prompt echo

Da tung gap truong hop model day prompt len lam cau tra loi. Nguyen nhan co the:

- Dung `/api/generate` theo prompt-completion, model chat-instruct co the echo prompt.
- Prompt qua dai va co nhieu rule.
- Stop/format khong khop chat template.
- Stream token dau la prompt echo, frontend hien ra nhu answer.

Huong da lam:

- Them `looks_like_prompt_echo`.
- Buffer stream ban dau de chan prompt echo.
- Neu RAG khong co output hop le, fallback ve cau `khong tim thay/no answer`.

Huong nen lam tiep:

- Chuyen sang `/api/chat` voi messages neu model Ollama ho tro tot.
- Giam prompt policy trung lap.
- Dung schema ro hon cho system/user/context.
- Log raw model output khi debug mode.

## 12. Frontend debug hien tai

Frontend da co:

- Hien route/debug pipeline.
- Hien sources.
- Hien score label:
  - dense
  - keyword
  - question
  - overview
- Nut download debug tam thoi.

File debug download nen chua:

- user question
- assistant answer
- route used
- route decision
- query intent
- catalog
- probe
- sources
- source previews

Muc dich:

- Nhanh tai ve case loi.
- So sanh cau hoi/cau tra loi/debug.
- Dung lam regression case cho pipeline.

Van de frontend con ton tai:

- Source label con kho doc.
- Score `keyword 2.0`/`dense 0.49` chua noi ro y nghia.
- Can hien `unit_type`, `chunk_reason`, `passage_number`, `question_range`, `parent_id` ro hon.
- Can co debug view rieng cho:
  - final context gui vao LLM
  - intent
  - source filtering
  - parent expansion
  - prompt echo/fallback

## 13. Nhung gi da trien khai

### 13.1 Retrieval probe + router context

Router khong con quyet dinh trong chan khong. Truoc khi route, backend lay:

- document catalog
- retrieval probe
- top hits
- score tung loai

Tac dung:

- Giam truong hop hoi tai lieu nhung bi route direct.
- Debug duoc router co thay tai lieu hay khong.

### 13.2 Structured IELTS parser/chunker

Da them parser/chunker de tao:

- `document_outline`
- `passage`
- `question_group`
- `question`

Tac dung:

- Query `Questions 1-4` tim duoc group/cau hoi dung hon.
- Overview co outline.
- Debug co `unit_type`, `passage_number`, `question_range`.

### 13.3 Overview retrieval

Da them `store.overview(...)`.

Tac dung:

- Query tong quan khong con phu thuoc top-k dense binh thuong.
- Co the lay outline + moi passage mot chunk dai dien.

### 13.4 Intent policy

Da tach intent:

- show
- translate
- explain
- solve
- overview
- semantic QA

Tac dung:

- Khong de mot prompt chung xu ly moi tinh huong.
- Solve moi duoc phep dua dap an.
- Show/translate/explain co policy cam giai.

### 13.5 Native-OCR reconciliation

Da co lop reconciler de giam duplicate native/OCR.

Tac dung:

- Giam chunk trung lap.
- Giam nhieu context.
- Giu alternative source de debug.

### 13.6 OCR theo huong PaddleOCR

Da uu tien PaddleOCR va warmup model.

Tac dung:

- Phu hop huong muon thay Tesseract bang PaddleOCR.
- Tesseract chi nen la fallback, khong phai main OCR.

### 13.7 Streaming va fallback

Da co:

- stream event
- metadata/debug event
- token event
- done/error event
- fallback khi model khong tra output hop le
- prompt echo guard

### 13.8 Debug export tren frontend

Da them debug export tam thoi de lay nhanh:

- question
- answer
- debug
- sources
- previews

## 14. Van de hien tai can review

### 14.1 Parser van la heuristic

Day la van de lon nhat.

Parser hien tai co cai tien, nhung van dua vao pattern/rule. Rule la can thiet o muc nao do, nhung neu bo sung qua nhieu rule cu the thi se overfit.

Rui ro:

- Tai lieu IELTS format khac se bi tach passage sai.
- Title co dau cham than/colon/line break la co the sai.
- Passage bi cat nham khi co heading/subheading giong pattern.
- Question range bi nham neu OCR sai so.
- Writing task/noise section bi parser gan nham vao passage.

Huong dung:

- Dung rule tong quat theo cau truc.
- Dung scoring + validator.
- Dung layout/bbox/font/page signal khi co.
- Luu diagnostics khi parser khong chac.
- Khong sua bang cach them topic-specific rule.

### 14.2 Table va flowchart con yeu

Hien question group dang co type:

- table completion
- flowchart completion

nhung noi dung table/flowchart van chu yeu la text linearized.

Rui ro:

- Mat cot/dong.
- Blank bi tach khoi question number.
- Flowchart mat quan he node-edge.
- OCR doc sai thu tu.

Huong tiep:

- Them element type `table` va `flowchart`.
- Dung PP-StructureV3 theo region can thiet.
- Luu bbox/crop/confidence.
- Khi table/flowchart fail, debug phai noi ro.

### 14.3 Intent van pattern-based

Intent hien tach ro hon, nhung van dung pattern cho action verbs.

Rui ro:

- User noi tu khac voi pattern thi route sai.
- `tra loi question 1-4` co the la solve, nhung `noi dung question 1-4` la show.
- `giai thich cau hoi` khac `giai cau hoi`.
- Tieng Viet co nhieu cach dien dat.

Huong tiep:

- Them intent classifier nhe co confidence.
- Dung deterministic parse cho question range.
- Dung safety default:
  - ambiguous + question range: show/explain, khong solve.
  - solve chi khi co solve intent ro.
- Log intent reason.

### 14.4 Overview intent chua bao phu du

Mot so cau nhu:

```text
Tai lieu nay gom nhung passage nao va moi passage co nhom cau hoi nao?
```

co the bi classify la `semantic_qa` thay vi `document_overview`, du van lay outline va tra dung.

Huong tiep:

- Them nhom intent `structured_overview`.
- Khong chi dua vao exact phrase `noi dung tai lieu`.
- Neu query hoi `gom passage nao`, `nhom cau hoi nao`, `outline`, `cau truc de`, route thang structured overview.

### 14.5 LLM van co the tu giai khi khong nen

Dac biet voi user hoi:

```text
Liet ke noi dung Questions 1-4
```

Mong muon:

- Liet ke cau hoi.
- Co the dich nghia ngan.
- Khong dua answer.
- Khong giai thich evidence.

Van de:

- LLM tutor co xu huong giai luon.
- Neu context co passage evidence, LLM se dung de giai.

Huong tiep:

- Voi `show_questions`, context chi chua question chunks.
- Them output validation.
- Neu can chinh xac tuyet doi, renderer deterministic lay structured data, sau do LLM chi "lam muot" voi policy khong giai.

### 14.6 Prompt echo/fallback

Da co guard nhung can tiep tuc theo doi.

Rui ro:

- Model khong stream token hop le.
- Guard chan nham output that.
- Frontend hien fallback khong du thong tin.

Huong tiep:

- Dung `/api/chat`.
- Luu raw output trong debug mode.
- Tach system/context/user thanh messages.

### 14.7 `documents.json` can rebuild khi schema doi

Moi lan thay parser/chunker/schema, index cu co the khong co metadata moi.

Rui ro:

- Frontend debug hien chunks cu.
- Retrieval khong dung `unit_type`.
- Question range search fail.

Huong tiep:

- Them `parser_version`/`chunker_version`/`schema_version`.
- Neu version mismatch, can rebuild.
- Co endpoint/admin action clear/rebuild index.

### 14.8 Chua co multi-user/session isolation

Hien store local co the dung chung:

```text
data/rag/documents.json
data/rag/embeddings.npy
```

Rui ro:

- File user nay anh huong user khac.
- Chat session khong tach docs.
- Khi upload file cung ten, upsert xoa/replace theo source_file.

Huong tiep:

- Them session_id/user_id/document_collection_id.
- Metadata filter theo session.
- Storage theo collection.

### 14.9 Debug extraction chua du sau

Can debug duoc:

- page route
- native quality
- OCR quality
- duplicates removed
- native/OCR contribution
- passages_found
- questions_found
- missing_questions
- duplicate_questions
- unassigned_questions
- chunk_reason
- final context sent to LLM

Hien frontend da co mot phan, nhung chua du de soi tat ca loi extraction/chunking.

## 15. Huong dang trien khai tiep

### Phase A: Lam vung parser boundary

Muc tieu:

- Khong overfit theo file IZONE.
- Khong hard-code title/topic/page/range.
- Giam passage/question boundary sai.

Viec can lam:

1. Them validator cho `IELTSDocument`.
2. Bao cao:
   - passage count
   - question count
   - missing question numbers
   - duplicate question numbers
   - unassigned question groups
   - suspicious passage boundaries
3. Dung layout signal neu extractor co bbox/font.
4. Tao regression set nhieu tai lieu IELTS khac nhau.

### Phase B: Structured index

Hien question search van nam trong `LocalVectorStore`.

Can tach structured index:

```json
{
  "questions": {
    "18": {
      "document_id": "...",
      "passage_number": 2,
      "group_id": "questions-18-23",
      "question_type": "table_completion",
      "text": "..."
    }
  },
  "passages": {
    "2": {
      "title": "...",
      "question_groups": [[14, 17], [18, 23], [24, 26]]
    }
  }
}
```

Dung structured lookup cho:

- cau 1 la gi
- questions 18-23
- passage 2 gom cau nao
- cau 20 thuoc passage nao
- dich cau 5

Dung semantic retrieval cho:

- vi sao dap an cau 20 la ...
- bang chung trong passage cho cau 5
- passage noi gi ve ...
- so sanh hai y trong tai lieu

### Phase C: Parent-child retrieval

Can search child chunk nho nhung dua parent context lon hon.

```text
Child hit:
    question-20

Parent context:
    questions-18-23

Evidence:
    passage-2 relevant paragraphs
```

Muc tieu:

- Giam context thua.
- Khong mat instruction.
- Co bang chung khi solve.
- Khong dua passage evidence khi chi show question.

### Phase D: Table/flowchart

Can them schema rieng:

```json
{
  "type": "table",
  "question_range": [18, 23],
  "columns": [],
  "rows": [],
  "bbox": [],
  "confidence": 0.89
}
```

```json
{
  "type": "flowchart",
  "question_range": [31, 35],
  "nodes": [],
  "edges": []
}
```

Khong nen dua table/flowchart thanh paragraph binh thuong.

### Phase E: Intent confidence + output validation

Can log:

- intent
- reason
- confidence
- action policy

Can validate output:

- `show_questions`: khong co answer labels.
- `translate_questions`: khong co answer labels.
- `explain_questions`: khong co answer labels.
- `solve_questions`: phai co evidence hoac noi thieu evidence.

### Phase F: Debug UI sau hon

Nen them tab:

- Route
- Intent
- Catalog
- Probe
- Sources
- Final context
- Extraction report
- Structure tree
- Chunk list

Va export mot file JSON gom full trace.

## 16. Nguyen tac khong overfit

Day la phan quan trong nhat cho cac buoc sua tiep.

Khong nen:

- Hard-code theo `IZONE _ IELTS READING TEST 2.pdf`.
- Hard-code title cu the nhu `Make That Wine!`, `Destination Mars`.
- Hard-code topic nhu `wine`, `Mars`, `Australia`.
- Hard-code page number.
- Hard-code rang passage 1 la 1-13, passage 2 la 14-26, passage 3 la 27-40.
- Hard-code dap an.
- Them rule chi dung vi mot OCR artifact cua mot file.

Nen:

- Dua vao pattern cau truc IELTS pho bien.
- Dua vao sequence cau hoi.
- Dua vao page/element order.
- Dua vao bbox/layout/font khi co.
- Dua vao validator va confidence.
- Neu khong chac, danh dau debug thay vi doan.
- Test tren nhieu file khac nhau.
- Moi heuristic phai co ly do va metadata/debug.

## 17. Cach test chatbot de kiem tra chat luong

### 17.1 Test overview

Hoi:

```text
Noi dung cua tai lieu tren la gi?
```

Ky vong:

- `query_intent=document_overview`
- source co `document_outline`
- co passage representative cho moi passage
- cau tra loi tom tat tat ca passage visible

Hoi:

```text
Tai lieu nay gom nhung passage nao va moi passage co nhom cau hoi nao?
```

Ky vong:

- Nen route structured overview/document overview.
- Tra ra passage title + question groups.
- Khong giai cau hoi.

### 17.2 Test show questions

Hoi:

```text
Liet ke noi dung Questions 1-4 trong tai lieu.
```

Ky vong:

- `query_intent=show_questions`
- source uu tien `question_group` va `question`
- chi liet ke instruction/statements
- khong co TRUE/FALSE/NOT GIVEN answer
- khong co passage evidence

### 17.3 Test explain questions

Hoi:

```text
Giai thich yeu cau cua Questions 1-4.
```

Ky vong:

- `query_intent=explain_questions`
- giai thich dang bai va cach hieu cau hoi
- khong dua dap an

### 17.4 Test solve questions

Hoi:

```text
Tra loi Questions 1-4 trong tai lieu.
```

Ky vong:

- `query_intent=solve_questions`
- source gom question group + passage context
- moi cau co answer + evidence ngan
- neu khong co evidence thi noi khong du evidence

### 17.5 Test negative semantic QA

Hoi:

```text
Trong tai lieu co noi ve mang xa hoi va sinh vien hoc tieng Anh khong?
```

Ky vong:

- Neu tai lieu khong co noi dung do, model phai noi khong tim thay trong tai lieu.
- Khong tra loi chung chung ve IELTS.
- Debug phai cho thay sources khong du manh hoac fallback no-match.

## 18. Ket luan trang thai hien tai

Pipeline hien tai da chuyen tu RAG text phang sang RAG co cau truc ban dau:

- Co ingestion da dinh dang.
- Co native/OCR reconciliation.
- Co IELTS parser.
- Co structured chunks.
- Co overview retrieval.
- Co question range retrieval.
- Co intent policy.
- Co parent passage expansion cho solve.
- Co debug export tren frontend.

Nhung van chua phai pipeline hoan chinh:

- Parser con heuristic.
- Intent con pattern-based.
- Table/flowchart con yeu.
- Overview intent can mo rong.
- Prompt echo/model blank output can theo doi.
- Need rebuild/versioning cho index.
- Chua co session isolation.
- Debug extraction/chunking/retrieval chua du sau.

Huong tiep theo nen tap trung vao:

1. Lam vung parser boundary theo huong tong quat, khong overfit.
2. Them structured index cho passage/question.
3. Tach exact lookup va semantic retrieval.
4. Cai tien generation policy va output validation.
5. Them table/flowchart element.
6. Tao regression set nhieu tai lieu de kiem tra chat luong.

