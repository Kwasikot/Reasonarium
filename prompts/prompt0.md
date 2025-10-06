# 🏛 Reasonarium

Твоя задача написать приложение

**Reasonarium** — desktop-приложение на базе **PyQt + Ollama + OpenAI**  
для тренировки рациональности, критического мышления и философской рефлексии.  

Мы вдохновляемся античными школами, еврейской традицией дебатов и современными когнитивными науками.  
Задача приложения — не давать готовые ответы, а раскрывать мышление человека через вопросы, споры и упражнения.  

---


# GUI Интерфейс
Я хочу видеть интерфейс похожий на интерфейс chatGPT с возможностью выбора промта. 
Он работает в стриминговом режиме с сервером openai.


## 🧩 Архитектура

- **PyQt** — графический интерфейс.  
- **Ollama** — локальные модели для автономного режима.  
- **OpenAI API** — облачные модели для сложных задач.  
- **Модули Reasonarium**:
  - `rationality_drive_node.py` — игра в рациональность  
  - `curiosity_drive_node.py` — генератор терминов и вопросов  
  - `debate_opponent_node.py` — споры (virtual/aggressive)  
  - `reflection_node.py` — философский интервьюер  
  - `tech_reflection_node.py` — обсуждение технологий  

В каталоге UsefulPrompts уже есть готовые промты, которые можно использовать.
UsefulPrompts\virtual_opponent_gpt_prompt.MD
UsefulPrompts\Prompt for ChatGPT — Aggressive Debate Opponent.MD
UsefulPrompts\Philosophical_Prompt_AI_Reflection0.md
UsefulPrompts\Philosophical_Prompt_AI_Reflection+Tutor.md
UsefulPrompts\Philosophical_Prompt_AI_Reflection.md
UsefulPrompts\gpt5_prompt_rationality_drive_v3.md
UsefulPrompts\gpt5_prompt_rationality_drive_v2.md
UsefulPrompts\gpt5_prompt_rationality_drive.md
UsefulPrompts\format_prompt_after_interview.md
UsefulPrompts\curios_child_prompt1.md
UsefulPrompts\constructive_criticism.md
UsefulPrompts\concept_rationality_drive.md
  

Также в корне проекта есть готовые модули rare_terms_node.py, curiosity_drive_node.py, know_you.py.
Реши сам как их можно использовать в контексте проекта.
В каталоге llm лежат интерфейсы для работы с llm. 
Но доработай их так чтобы они могли работать в стриминговом режиме, как веб интерфейс chatGPT.
---

## 🔑 Основные модули

### 🎲 Rationality Drive
Интерактивная игра, где пользователь сталкивается с когнитивными ловушками и вероятностными задачами.  
- выбор решений,  
- немедленная обратная связь,  
- **очки рациональности** и уровни: от *Naïve Thinker* до *Master of Clear Thinking*.  

---

### 🧘 Philosophical Reflection
Режим личного интервьюера:  
- задаёт один глубокий вопрос в день (о сознании, будущем, этике, роли AI),  
- ждёт развернутого ответа,  
- при необходимости даёт контраргумент или уточнение.  

---

### ⚔️ Virtual Opponent
Классический режим дебатов:  
- на каждый ответ → **контраргумент + уточняющий вопрос**,  
- помогает видеть когнитивные ошибки и скрытые допущения.  

---

### 🥶 Aggressive Opponent
Интенсивный тренажёр:  
- «двойной удар» (острый вопрос + жёсткий контраргумент),  
- быстрый ритм, ирония, логическое давление,  
- развивает устойчивость и точность аргументации.  

---

### 🔬 Curiosity Drive
Возможна как отдельная GUI форма.
Модуль борьбы с эффектом Даннинга–Крюгера.  
- Выбирает случайную дисциплину и подтему.  
- Предлагает **редкий термин** с коротким объяснением.  
- Задаёт **вопрос по этому термину**.  
- Генерирует **поисковый запрос для YouTube**, чтобы пользователь мог изучить тему глубже.  
- Ведёт человека от ложной уверенности к более реалистичному и зрелому пониманию.  
- Также может рекомендовать литературу.
---

### ⚙️ Technical Reflection
Режим обсуждения технологий и приборов:  
- берёт описание из книги/статьи,  
- кратко пересказывает (3–6 предложений),  
- задаёт вопрос о плюсах и минусах,  
- отвечает минимально, чтобы **раскрыть мышление собеседника**, а не дать готовый список.  

---


### 🧪 Popper Challenge (новый модуль)
Экспериментальный режим по критерию научности Карла Поппера.  
- ИИ **синтезирует теорию**, объясняющую какое-либо явление.  
- Задача пользователя — **найти опровержение** или слабое место.  
- Критерий: теория считается научной только если её можно **фальсифицировать**.  
- Пользователь получает очки за способность:  
  - сформулировать тестируемое предсказание,  
  - предложить возможный эксперимент или наблюдение, которое опровергло бы теорию,  
  - указать на нефальсифицируемые утверждения (и тем самым выявить псевдонауку).  

Этот модуль тренирует умение отличать **науку от ненауки**, мыслить скептически и использовать инструмент Поппера в практике.  

---

## Out-of-the-box thinking 
Можно встроить как бустер креати#вности:
Генератор неожиданных связок (например, термин из биологии + задача из экономики).
Вопросы «а что если наоборот?» после стандартного ответа.
Тренажёр: придумай 5 нестандартных применений для обычного объекта.
Очки «креативности» в дополнение к очкам рациональности.

## Tech Skeptic Mode
Она будет называться "Tech Skeptic Mode".
У этого тренера цель — учить критиковать описания техники и идей, находить слабые места, ложные обещания, псевдонаучные элементы. То есть он развивает инженерный скептицизм и навык «видеть баги» в красивых описаниях технологий.
Я хочу чтобы модель синтезировала короткое описание вымышленного устройства или технологии в разных областях (по спектру дисциплин MIT), и в тексте уже были встроены слабые места для критики. Пользователь затем тренируется находить ошибки, логические дыры, преувеличения, нефальсифицируемые утверждения.

## Опровержение гипотезы
Этот режим поможет развить аналитические способности, например для докторов.
Игра в опровержение гипотезы. 
ИИ предлагает некоторую игровую ситуацию и формирует гипотезу.
Потом он дает новые свидетельства, и ты как в байесовском стиле стараешься обновить вес гипотезы.
Короче тебе нужно учиться быстро опровергнуть некоторую гипотезу, принимая во внимание все детали этой мини истории.

## 🖼 Интерфейс (PyQt)

- Чат-подобный UI с историей.  Ты можешь разработать универсальный компонент.
- Переключение режимов одной кнопкой.  
- Панель «Очки рациональности» с графиками прогресса.  
- Вкладка «Знаете ли вы что…» для случайных фактов и терминов.  
- Поддержка локального (Ollama) и облачного (OpenAI) движка.  


---

---

## README.MD
Создай файл requirements.txt со списком pip пакетов для установки.
Напиши ридми файл. Перечисли там команды для запуска и настройки пайтон среды.
Ориентируемся на Python 3.11

---
Сделай настройки языка в программе.
Пример файла настройки в settings/reasonarium_settings.xml.
1. Тебе нужно перевести все промты на все языки.
2. Сделать выпадающий combobox с выбором языка.
3. Переводиться должны как промты так и все виджеты в главном окне.
4. Сделай отдельную вкладку с формой для такой функции (Curiosity Drive)
  def _curiosity_fallback_json(self, disciplines, audience, rarity, novelty, n):
        # deterministic simple filler
        import random
        if not disciplines: disciplines = ['General Science']
        picked = random.choice(disciplines)
        meta = {"audience": audience, "rarity": rarity, "novelty": novelty, "discipline_pool": disciplines, "picked_discipline": picked, "n": n, "timestamp": datetime.datetime.now().isoformat()}
        items = []
        for i in range(n):
            items.append({
                'concept': f'{picked} concept {i+1}',
                'rare_term': None,
                'kid_gloss': f'A short explanation for item {i+1}',
                'hook_question': f'What if {picked} {i+1}?',
                'mini_task': f'Try a small experiment {i+1}',
                'yt_query': f'{picked} intro'
            })
        return {'meta': meta, 'items': items}

Промт и список дисциплин  возьми из файла reasonarium_settings.xml
---------------------------------------
1. Убери комбобокс выбора Mode.
2. Когда я выбираю из комбомокса промт Curisosity Drive грузится промт concept_rationality_drive.md. Назови Rationality Drive.
---------------------------------------
В режиме Rationality Drive не надо спрашивать тему.
---------------------------------------
Сделай выбор модели как комбобокс возьми список из reasonarium_settings.xml
-------------------------
Выделяй отдельным цветом текст
A: Контраргумент

и вопрос например зеленым
текст
B: 

Отправляй сообщение по ctrl+enter
-------------------------
В режиме Engine: ollama сделай speech to text через whisper.
-------------------------
Доработай диалог выбора темы для дебатов.
Сделай два комбобокса disciplines и subtopics.
Соответсвующие массивы возьми из curiosity_drive_node.py и алгоритм генерации subtopics.
После выбора discipline и subtopic он делает обращение к LLM с таким промтом (на языке текущего выбранного в программе языка).
Prompt:
"Выбери наиболее спорные вопросы по дисциплине {discipline} и подтеме {subtopic}. Выбери ровно 20 вопросов."
По нажатию кнопки ok диалога заданный вопрос попадает дальше в режим спора.

-------------------------
Диалог выбора темы должен быть шире.
Там должне быть spinBox для выбора количества выбираемых вопросов.
И отдельное поле для кастомного пользовательского вопроса.
-------------------------
После кнопки "New Chat" добавь кнопку "Auto answer".
По нажатию кнопки LLM сама предлагает наиболее адекватный ответ на вопрос.
-------------------------
Popper Challenge (новая вкладка)
Экспериментальный режим по критерию научности Карла Поппера.
Сделай два комбобокс disciplines.
Соответсвующие массив возьми из curiosity_drive_node.py.
Сделай кнопку "Синтезировать теорию".
ИИ синтезирует теорию по выбранной дисциплине, объясняющую какое-либо явление.
Задача пользователя — найти опровержение или слабое место.
Критерий: теория считается научной только если её можно фальсифицировать.
Потом LLM проверяет это опровержение.
Пользователь получает очки за способность:
сформулировать тестируемое предсказание,
предложить возможный эксперимент или наблюдение, которое опровергло бы теорию,
указать на нефальсифицируемые утверждения (и тем самым выявить псевдонауку).

Замени промт на такой
Prompt:
Synthesize a short science-oriented theory (3–6 sentences) in a randomly chosen domain ({{d1}{' and ' + d2 if d2 else ''}}). The theory should not repeat previous themes (avoid fruits, trees, or overly narrow motifs).

Then add three sections:
A) Predictions — at least 2 clear, testable predictions derived from the theory.
B) Experiments/Observations — possible ways to falsify these predictions.
C) Unfalsifiable — identify any parts of the theory that cannot be tested, and explain why that is problematic.

The theory may be serious, playful, whimsical, or absurd — but it must still follow Popper’s criterion of scientific testability.
-------------------------
Добавь в диалог Popper Challange выбор subtopic.
Соответсвующие массивы возьми из curiosity_drive_node.py и алгоритм генерации subtopics.
Прокинь этот subtopic в промт.
--------------------------
Мы можем ввести две шкалы сложности:

По типу теории (от абсурдной до продвинутой кросс-дисциплинарной).

По уровню образования (школа → университет → магистратура → аспирантура/докторат).

Тогда можно комбинировать: например, «простая абсурдная теория на школьном уровне» или «кросс-дисциплинарная научная теория на уровне аспирантуры».

🔹 Уровни сложности по типу теории (добавь combobox в диалоге Popper Challenge)

Trivial / Absurd — намеренно простое или сказочное объяснение.

Folk / Intuitive — бытовое или «народное» объяснение.

Speculative / Pseudoscientific — правдоподобно звучащее, но с логическими дырами.

Scientific-Style — похоже на научную гипотезу.

Advanced / Cross-disciplinary — сложная, с привлечением нескольких областей.

🔹 Уровни сложности по образованию (добавь combobox в диалоге Popper Challenge)

School — язык максимально простой, базовые понятия (атомы, клетки, планеты).

Undergraduate (Bachelor) — более сложные концепции, базовые модели науки.

Graduate (Master) — углублённые рассуждения, междисциплинарные связи.

Doctoral / PhD — очень высокая детализация, профессиональная терминология, отсылки к реальным теориям.

Удостоверься что все строки переведены на язык выбранный в программе.
--------------------------
1. сделай так чтобы 'Эксперименты/Наблюдения' предлагал пользователь, и опционально по checkbox можно было бы посмотреть что предложил ИИ. 
2. Добавь кнопку "Подтвердить эксперимент" по которой вызывается промт "Give me most brutally honest constructive criticism you can" для проверки валидности эксперимента в рамках предложенных предсказаний теории. 
3. Сделай как обычно все строки и промты на выбранном в программе языке.
--------------------------
Там где кнопка "Brutal critics" добавь еще одну кнопку "Оцени мою критику".
Со следующей логикой:
1. При нажатии на эту кнопку LLM оценивает критику сгенерированной теории как способность человека критически мыслить.
Признавай сильные стороны критики, но ищи неочевидные и очевидные уязвимости. 
Опиши плюсы и минусы его критики (отдельным загаловком)
Что не хватает в его критике (под отдельным загаловком).
Обнаружь когнитивные искажения (отдельным загаловком).
Обнаруживая потенциальные проблемы в мышлении с психиатрической точки зрения (отдельным загаловком).

2. Переименуй лейбл "Ваши эксперименты \ наблюдения" в "Ваши эксперименты \ наблюдения \ критика"
3. Переименую кнопку "Жесткая критика" в "Жесткая ИИ критика"
4. Сделай все строки и промты на двух языках.
--------------------------
Во вкладке Poppers Challenge на всех кнопках использующих LLM смотри чтобы при вызове ЛЛМ текст отображался в стриминговом режиме.
Так чтобы было понятно что ЛЛМ генерирует токены. 
--------------------------
В функции on_popper_eval_selfcrit перепиши промты так чтобы они были не для двух языков, а для произвольного языка. 
Просто ключи в сам промт язык как переменную, чтобы вывод был на языке интерфейса.
---------------------------
--------------------------
Добавь новую вкладку сразу после Poppers Challenge.
Она будет называться "Tech Skeptic Mode".
У этого тренера цель — учить критиковать описания техники и идей, находить слабые места, ложные обещания, псевдонаучные элементы. То есть он развивает инженерный скептицизм и навык «видеть баги» в красивых описаниях технологий.
Я хочу чтобы модель синтезировала короткое описание вымышленного устройства или технологии в разных областях (по спектру дисциплин MIT), и в тексте уже были встроены слабые места для критики. Пользователь затем тренируется находить ошибки, логические дыры, преувеличения, нефальсифицируемые утверждения.

Интерфейс этой вкладки состоит из текстового поля для синтезированного описания, текстового поля для критики.
Кнопки "Синтезировать описание технологии", "Analyze my criticism", "Жесткая ИИ критика технологии".



Шаблон Synth Prompt (вызывается по кнопке "Синтезировать описание технологии"):
"""
Generate a short description (5–8 sentences) of a **fictional device or technology** 
in a randomly chosen discipline ({discipline}). 

The description must follow the chosen **Education Level**: 
- School = simple and intuitive explanation 
- Undergraduate = moderately technical 
- Graduate = advanced and interdisciplinary 
- PhD = highly technical, jargon-heavy, with references to theories. 

Important: The description should intentionally include at least 2–3 weak spots 
(logical flaws, vague claims, overgeneralizations, or unfalsifiable parts) so that 
the user can practice criticizing it. 
Make the flaws subtle at higher levels and obvious at lower levels. 

Output format:
1) Title of the device/technology
2) Description (5–8 sentences, with hidden weaknesses)
"""

Шаблон Prompt 1 (вызывается по кнопке "Analyze my criticism"):
"""
You are evaluating a user's critique of a fictional device/technology description. 
The critique should be judged on 5 criteria: 
1) Did the user correctly identify weak points in the text? 
2) Did they point to specific statements or assumptions? 
3) Did they propose testable ways to falsify or check the claims? 
4) Is their reasoning clear and logically structured? 
5) Does the critique match the education level of the original text (School → PhD)? 

Output format:
- Score: give a score from 1 to 10.
- Strengths: list 2–3 things the user did well.
- Weaknesses: list 2–3 areas for improvement.
- Suggestion: one short tip to improve the critique next time.

User critique:
{user_critique}
"""

Шаблон Prompt 2(вызывается по кнопке "Жесткая ИИ критика технологии")
"""
You are a critical technology evaluator . 
Given a description of a fictional device or technology, produce a **rigorous, constructive, and uncompromising critique**.  

Your goals:  
1. Identify **pros (strengths)**: any plausible, well-explained, or innovative aspects.  
2. Identify **cons (weaknesses)**: hidden assumptions, logical flaws, vague or exaggerated claims, contradictions, engineering impossibilities, or unfalsifiable parts.  
3. Suggest **ways the device could be tested, improved, or reformulated**.  
4. Maintain a tough but professional tone: highlight the most serious flaws without softening the judgment.  

Format:  
**Pros**: (bullet list, concise but insightful)  
**Cons**: (bullet list, focused on weak points and potential failures)  
**Recommendations**: (short, constructive advice for improvement)  

Technology description:
{tech_description}
"""

🔹 Уровни сложности (по образованию) (отдельный комбобокс наверху)
1. School: простое объяснение, метафоры, базовые понятия.
2. Undergraduate: чуть глубже, базовые термины из инженерии/наук.
3. Graduate: технически сложнее, упоминания моделей и эффектов.
4. PhD: насыщенный жаргоном, отсылки к теориям и методологиям, но намеренно с «слабыми местами».

Список дисциплин возьми из tech_disciplines.md.

Промты Synth Prompt, Prompt 1, Prompt 2 должны содержать переменную языка выбранного в интерфейсе программы.
Строки элементов интерфейса вкладки "Tech Skeptic Mode" должны быть переведены и записаны в reasonarium_settings.xml

-----------------
Сделай чтобы центральную часть внутри вкладки "Popper Challenge": виджет Your experiments... чек бокс "Show all suggestions", кнопки "Confirm" "Evalute.." "Brutal AI" можно было прятать. Чтобы увеличить размер нижнего виджета Evalution. Предложи как лучше сделать, может быть через сплиттер?
-------------------
Переделай промт в Evaluate falsification так чтобы он только проверял только опровергают ли предложенные эксперименты теорию и если пользователь ввел не описание экспремента (или описание какого-то другого эксперимента, который не подходит под рамки теории) выдавай что ты не можешь принять данное описание эксперимента. Оценки можешь не выставлять. Просто посчитай количество валидных экспериментов.
---------------------
Да и переделай этот промт так чтобы он как и другие промты выдавал ответ в языке интерфейса (Respond strictly in {lang_name}) 
---------------------
Добавь в reasonarium_settings.xml в список языков 100 наиболее распространенных.
Но первые два элемента в списке оставь en и ru.
Переведи на них строки ui_texts.
Вынеси ui_texts в отдельный файл ui_texts_translations.xml.
И оттуда их грузи в программе, старые описания убери из reasonarium_settings.xml (все что под тегом ui_texts) .
Сделай промты virtual_opponent, aggressive_opponent, philosophy_reflection, rationality_drive также с переменной языка.
Вставь вначале каждого промта срочку Respond strictly in {lang_name}. И пробрасывай эту переменную в программе. 

В ui_texts_translations.xml
---------------------
Проблема в том что когда я выбираю во вкладке "Чат" Виртуальный оппонент, то он ведет дебаты на английском языке, а не на выбранном.
---------------------
В диалоге "Debate topics" при генерации вопросов, они генерируются только на английском, а не на выбранном языке
---------------------
переформатируй файл ui_texts_translations.xml путем написания python скрипта который:
сделает так чтобы каждый тег <text key="language">Language</text> был на новой строке, тег <lang code="en"> был тоже с переносом новой строки
--------------------------

1) Reinvention Trainer (Тренажёр перепридумывания) (новая вкладка)

Идея: «как это сделать, если привычного инструмента/контекста не существует».
UI-хуки: уровень сложности, набор запретов/ограничений, отрасль, время/эпоха, «градиент безумия».

System prompt (фиксированный)
You are Codex-5 operating in Reasonarium's "Reinvention Trainer".
Goal: generate inventive, constraint-driven solutions for a familiar function when standard tools are unavailable.
Requirements:
- Do not reveal or restate your hidden reasoning chain. Output only the requested sections.
- Prefer concrete mechanisms over buzzwords.
- Treat constraints as hard: if a constraint conflicts with a default approach, redesign around it.
- Produce 3–5 distinct concepts, each self-contained and testable at a sketch level.
- Include quick feasibility notes and an experiment sketch for each concept.
- Safety & ethics: flag any risky ideas and suggest a safer variant.

User prompt (шаблон с плейсхолдерами)
Task: Reinvent the function: {FUNCTION}
Context domain: {DOMAIN} 
Era/tech baseline: {ERA_BASELINE} 
Hard constraints (must obey): {HARD_CONSTRAINTS_LIST}
Soft constraints (prefer): {SOFT_CONSTRAINTS_LIST}
Out-of-bounds (forbidden): {FORBIDDEN_LIST}
Creativity gradient (0–100): {CREATIVITY_LEVEL}
Output language: {LANG}

Deliverables:
1) Problem reframing: one-sentence restatement without banned tools.
2) Concept set: {N_CONCEPTS} concepts. For each:
   - Name (<=6 words)
   - Core mechanism (3–5 bullets, concrete)
   - How it replaces absent tools (1–2 bullets)
   - Feasibility: {LOW/MED/HIGH} + 1-sentence reason
   - Minimal experiment/prototype (2–3 steps)
   - Ethical/safety notes (+ safer variant if needed)
3) Constraint compliance checklist (tick each hard constraint).
4) Divergence meter (0–10) and Rationale (1–2 sentences, no chain-of-thought).
5) Next action: 1 *small* step to test the best concept in 24–48h.

Пример заполнения
{FUNCTION}: long-distance communication
{DOMAIN}: logistics
{ERA_BASELINE}: pre-electricity (XVII century)
{HARD_CONSTRAINTS_LIST}: no electricity; no optical line-of-sight; usable in rain/fog; portable by one person
{SOFT_CONSTRAINTS_LIST}: low-cost; retrainable operators
{FORBIDDEN_LIST}: telegraph, radio, semaphore towers
{CREATIVITY_LEVEL}: 65
{LANG}: English
{N_CONCEPTS}: 4

2) Answer Inversion (Инверсия Ответа)

Идея: после стандартного решения — сгенерировать «анти-решение»/обратную интерпретацию, проверить устойчивость аргументации и обнаружить слепые зоны.
UI-хуки: кнопка «Flip», ползунок «Radicality», чекбоксы «keep ethics», «stress test only».

System prompt (фиксированный)
You are Codex-5 in Reasonarium's "Answer Inversion".
Goal: generate a rigorous opposite take on a prior answer, then reconcile both views.
Rules:
- Do not expose hidden reasoning. Output only requested sections.
- Attack ideas, not people. Maintain ethical and factual integrity.
- If the original answer is unsafe or clearly false, do not mirror it—explain the issue and propose a safe inversion test.
- Use evidence-backed, concrete points. Avoid strawmen.

User prompt (шаблон)
Original question: {QUESTION}
Original (baseline) answer: {BASELINE_ANSWER}
Inversion radicality (0–100): {RADICALITY}
Hard ethical guardrails: {ETHICS_GUARDRAILS}
Context/domain: {DOMAIN}
Output language: {LANG}

Deliverables:
A) One-sentence steelman of the baseline answer.
B) Inverted thesis (1 sentence) aligned with {RADICALITY}.
C) Top-5 arguments for the inverted thesis (bullet list, concrete).
D) Stress-test table (3–5 rows):
   - Row fields: Claim | Weakest link | What would falsify it | Quick check to run this week
E) Synthesis: 3 statements where both sides can be simultaneously true (conditions/boundaries).
F) Decision hooks:
   - If you had to act tomorrow: 1 pragmatic choice and why (≤3 sentences).
   - What evidence would most change your mind (≤3 bullets).

Мини-пример
{QUESTION}: How to reduce urban traffic congestion?
{BASELINE_ANSWER}: Expand public transit and congestion pricing.
{RADICALITY}: 70
{ETHICS_GUARDRAILS}: no harm, no discrimination; privacy preserved
{DOMAIN}: urban policy
{LANG}: English

Формат ответа (оба режима) — единый JSON для Reasonarium

Так вы сможете хранить/оценивать/складывать очки:

{
  "mode": "reinvention|inversion",
  "meta": {
    "creativity": 65,
    "radicality": 70,
    "domain": "urban policy",
    "lang": "en"
  },
  "sections": {
    "reframing": "...",
    "concepts": [
      {
        "name": "...",
        "mechanism": ["..."],
        "replaces_absent_tools": ["..."],
        "feasibility": "MED",
        "experiment": ["step1", "step2"],
        "ethics": {"risks": ["..."], "safer_variant": "..."}
      }
    ],
    "checklist": [{"constraint": "...", "ok": true}],
    "divergence": {"score": 8, "rationale": "..."},
    "stress_table": [
      {"claim": "...", "weakest_link": "...", "falsify": "...", "quick_check": "..."}
    ],
    "synthesis": ["...", "...", "..."],
    "decision_hooks": {
      "act_tomorrow": "...",
      "evidence_to_change_mind": ["...", "..."]
    },
    "next_action": "..."
  },
  "scores": {
    "creativity_points": 0,
    "rationality_points": 0
  }
}

Авто-скoring (креативность + рациональность)

Creativity points (0–10):

+2 если ≥3 концепта различаются по принципу действия

+2 если выполнены все hard constraints

+2 если есть принципиально новый механизм (не просто комбинация)

+2 если эксперимент реально выполним без крупных бюджетов

+2 если есть «дивергенция ≥7» и аргументировано почему

Rationality points (0–10):

+3 за явные критерии фальсификации

+3 за этические оговорки и безопасный вариант

+2 за конкретные метрики успеха в эксперименте

+2 за «act tomorrow» с чётким trade-off

(Можете считать баллы на бэкенде, сверяясь с заполненностью полей.)

Параметры генерации (рекомендации)

Reinvention Trainer: temperature 0.9, top_p 0.9, presence_penalty 0.6

Answer Inversion: temperature 0.7, top_p 0.8, frequency_penalty 0.4

Для «радикальности/креативности» можно линейно маппить на temperature/presence_penalty.

Быстрые UI-паттерны

Кнопка Reinvent → открывает диалог с пресетом ограничений (чекбоксы «без электричества», «только биологические носители», «офлайн/air-gapped», «эпоха: средневековье/индустриальная»)

Кнопка Flip (Инвертировать ответ) рядом с любым сгенерированным решением. Ползунок Radicality и чекбокс «Этика прежде всего».

«📎 Эксперимент-превью»: мини-карточка с 2–3 шагами и метрикой успеха.

«✅ Checklist» рендерится как теги-бейджи, мгновенно видно, что соблюдено.

Мини-пример вызова (Reinvention Trainer)
FUNCTION: personal identity verification for payments
DOMAIN: fintech
ERA_BASELINE: no digital systems, paper allowed
HARD_CONSTRAINTS_LIST: no biometrics; no centralized registry; offline verification in <60s
SOFT_CONSTRAINTS_LIST: low training time for clerks
FORBIDDEN_LIST: passwords, OTPs
CREATIVITY_LEVEL: 55
LANG: English
N_CONCEPTS: 4

Дополнительные инструкции:
1. Не забудь что новая вкладка Reinvention Trainer поддерживала все языки доступные в Reasonarium. 
   Т.е. при выборе языка эта вкладка переводится как и все остальные.
----------
Мне не нравится что система по дефолту пользователю не предлагает никакие варианты, как допустим в Poppers Challenge есть кнопка синтезировать теорию которая сразу же заполняет поля. Здесь надо сделать что-то похожее. И не понятно зачем пользователю показывать какой-то там JSON. Не надо чтобы пользователь думал в какое поле что ему писать, для него должно быть одно поле ввода как в чате гпт, остальное должна заполнять машина.
 