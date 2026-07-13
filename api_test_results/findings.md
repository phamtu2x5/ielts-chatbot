# Batch Test Findings

API URL: `https://involvement-buzz-joseph-orlando.trycloudflare.com/api/chat`

Tong so cau test: `19`

Tat ca 19 request rieng le da tra HTTP `200`.

## File ket qua

- `test_01.json` den `test_19.json`: raw response + debug + sources + source previews.
- `test_01.md` den `test_19.md`: ban de doc nhanh theo tung cau.
- `summary.md`: bang tong hop route/intent/source_count/flags.
- `run_batch_tests.py`: script tao request va collect raw output.
- `request_01.json` den `request_19.json`: body request da gui.
- `raw_01.body` den `raw_19.body`: raw body API tra ve.

## Van de noi bat

### 1. Parser boundary cua Reading dang sai

Tat ca test deu flag:

```text
Suspicious passage_numbers for IZONE _ IELTS READING TEST 2.pdf: [1, 2, 3, 4, 5, 6]
```

Tai lieu IELTS Reading nay dang bi tach thanh 6 passage:

- Passage 1: `Make That Wine!`
- Passage 2: `Choose NO MORE`
- Passage 3: `That Vision Thing`
- Passage 4: `Choose NO MORE`
- Passage 5: `Destination Mars`
- Passage 6: `Choose NO MORE`

Trong khi ky vong hop ly hon la 3 passage chinh. Cac dong instruction nhu `Choose NO MORE...` dang bi parser nham thanh title/passage moi.

Anh huong:

- Overview sai.
- `Passage 2` bi lech.
- `Passage 3` bi lech thanh passage 5.
- Task/question group mapping khong on dinh.

File lien quan de xem:

- `test_01.md`
- `test_02.md`
- `test_03.md`
- `test_19.md`

### 2. Intent policy con nham "show table/flowchart" thanh solve

Test 08:

```text
Hiển thị lại toàn bộ bảng của Questions 5–10...
```

Bi detect:

```text
query_intent=solve_questions
```

Trong khi user noi ro:

```text
Không giải bài.
```

Ket qua model tu dien dap an vao bang, vi intent sai va context co passage evidence.

Test 09 cung bi:

```text
query_intent=solve_questions
```

Trong khi user chi muon hien thi cau truc flowchart, chua dien dap an.

File lien quan:

- `test_08.md`
- `test_09.md`

Huong sua:

- `không giải bài`, `chưa điền đáp án`, `chưa giải`, `không đưa đáp án` phai override solve markers.
- Them intent rieng: `show_table`, `show_flowchart`, hoac gom vao `show_questions`.
- Neu co phrase "hien thi / liet ke / cau truc / markdown / giu dung hang cot" thi khong solve.

### 3. Table/flowchart extraction chua du cau truc

Test 08 source cho Questions 5-10 chi co:

```text
Questions 5-10 Complete the table.
```

Khong co noi dung table that su trong `question_group` chunk. Model da tu suy dien/dien dap an tu passage, khong phai hien lai bang goc.

Test 09 yeu cau cau truc flowchart nhung model tao flowchart suy dien theo passage, khong dua cau truc visual that.

Huong sua:

- Them element/chunk rieng cho table va flowchart.
- OCR/layout can giu hang/cot/node/edge/blank/question number.
- Question group 5-10 va 18-23 phai co visual/text representation cua table/flowchart, khong chi instruction.

### 4. Writing image retrieval dang yeu

Cac test Writing:

- `test_04`: hoi yeu cau de Writing trong anh -> route `base_model`, khong dung RAG.
- `test_12`: hoi trich xuat bang trong anh -> route RAG nhung source lai la outline Reading, tra "cannot find".
- `test_13`: hoi smartphone ownership country B 2024 -> route direct, khong dung image chunk.
- `test_14`: hoi Internet Access tang lon nhat -> route direct, khong dung image chunk.
- `test_15`: viet overview -> route direct va hallucinate de moi ve moi truong.
- `test_16`: route RAG nhung retrieve sai sang Reading passage, khong lay image table.

Anh huong:

- Cac cau hoi co cum `ảnh`, `Writing`, `table`, `smartphone`, `Internet Access`, `Country B` chua duoc route/retrieve dung vao image document.

Huong sua:

- Document catalog nen co document type/summary cho image OCR.
- Image chunk can metadata:
  - `document_type=ielts_writing_task_1`
  - `unit_type=writing_prompt/table`
  - `table_columns`
  - `table_rows`
- Retrieval should boost image/table chunks khi query co:
  - `ảnh`
  - `writing`
  - `bảng`
  - `smartphone`
  - `internet access`
  - `country`
- Structured table lookup cho cell queries.

### 5. Negative document QA dang route direct va hallucinate

Test 17:

```text
Đề Reading có nhắc đến mạng xã hội và việc học tiếng Anh của sinh viên không?
```

Bi route:

```text
route_used=base_model
query_intent=direct
sources=0
```

Model tra loi chung chung ve social media/TikTok/YouTube, trong khi can dung RAG va noi khong tim thay trong Reading.

Huong sua:

- Query co `Đề Reading`, `tài liệu`, `trong Reading`, `trong file`, `có nhắc đến... không` phai la document intent.
- Negative QA can retrieve outline + relevant chunks; neu no-match thi answer grounded "khong tim thay trong tai lieu".

### 6. Cau hoi nham giua Reading va Writing can metadata filter

Test 18:

```text
Trong Reading Passage 1, tác giả nói rằng smartphone ownership tăng mạnh nhất ở Country C đúng không?
```

Ket qua kha tot: route RAG, lay Passage 1 va noi khong co smartphone/Country C trong Reading Passage 1.

Nhung source ranking van co question chunk xen vao truoc passage chunk.

Huong sua:

- Khi query co `Reading Passage 1`, filter passage_number=1 va unit_type=passage truoc.
- Sau do moi lay question/outline neu can.

### 7. Solve Questions 1-4 da chay duoc nhung evidence/explanation can chat hon

Test 07:

- route `vector_rag`
- intent `solve_questions`
- co question chunks + passage context

Nhung can xem lai chat luong dap an:

- Cau 3 dang tra `FALSE` voi ly do "khong de cap pho bien hien nay"; trong T/F/NG, neu khong co thong tin ve popularity in Near East thi thuong nen la `NOT GIVEN`, khong phai `FALSE`.
- Cau 4 dang `NOT GIVEN`; can doi chieu voi passage that su va expected answer.

Huong sua:

- Prompt solve T/F/NG can nhan manh:
  - contradicted -> FALSE
  - absent/unsupported causal relation -> NOT GIVEN
- Can trich evidence chinh xac hon, khong chi paraphrase.
- Can co answer validation bang rule T/F/NG neu evidence khong mau thuan truc tiep.

## Uu tien sua tiep

1. Parser boundary: chan instruction lines nhu `Choose NO MORE...` bi nham thanh passage title.
2. Intent policy: phrase phu dinh `không giải/chưa điền/không đưa đáp án` phai override solve intent.
3. Image/Writing retrieval: boost/filter image table chunks theo query ve Writing/anh/table/country/smartphone/internet.
4. Table/flowchart structure: extract va chunk visual structure, khong chi instruction.
5. Negative QA: document-intent detection cho "trong Reading/tài liệu có nhắc đến ... không".
6. T/F/NG solving: lam chat prompt/evidence logic de tranh FALSE vs NOT GIVEN sai.

