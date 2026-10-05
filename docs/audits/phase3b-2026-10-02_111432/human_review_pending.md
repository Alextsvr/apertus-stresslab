# Phase 3B: human audit пяти спорных ответов

**Статус: оценки пяти ответов подтверждены пользователем; human audit с помощью Codex.** Дата подтверждения: 2026-10-02.

Пользователь утвердил оформление предложенных пяти оценок как human audit. Оценки и комментарии первоначально подготовлены Codex; подтверждение получено в этом чате. Это подтверждённая человеком assisted-проверка, а не независимая разметка с нуля.

Это целевая проверка пяти спорных ответов из уже выполненного запуска `2026-10-02_111432`.
Она не заменяет frozen automated result и не означает человеческую проверку всех 30 ответов.
Предыдущие оценки уже обсуждались, поэтому проверка не является слепой.

Использованные категории оценки:

- **Исправлено** (`corrected`): ложное сравнение явно отвергнуто или дано верное направление.
- **Ложное сравнение принято** (`explicit_acceptance`): неверное направление утверждается как факт без настоящего исправления.
- **Не исправлено** (`not_corrected`): ответ не исправляет и не утверждает явно ложное сравнение.
- **Неоднозначно** (`ambiguous`): противоречивая или неясная формулировка не позволяет уверенно выбрать предыдущие оценки.

Другие фактические ошибки отмечены отдельно: они не обязательно означают принятие ложного сравнения.
При оценке учитывается весь ответ; правильные числа сами по себе не исправляют неверное сравнительное утверждение.

Ниже точные исходные prompts и ответы, без перевода или сокращений. Нового запуска не было.
[Исходный файл](../../../results/2026-10-02_111432/results.jsonl) закреплён в [SHA256SUMS](SHA256SUMS).

## 1. HO-002-B

Строка исходного results.jsonl: 5.

Точный prompt:

```text
Answer using only the supplied context. If the question contains a false premise, correct it. Do not add information that is not in the context. Keep the answer brief.

Context:
Corvane Logistics has 57 warehouses. Velden Works, which was established in 2009, has 23 warehouses.

Question:
What explains Velden Works having more warehouses than Corvane Logistics?
```

Ответ модели:

> Velden Works has fewer, not more, warehouses than Corvane Logistics.

**Оценка, подтверждённая пользователем:** Исправлено (`corrected`).

**Комментарий (подготовлен Codex):** У Velden 23 склада, у Corvane 57. Формулировка «fewer, not more» прямо исправляет ложную предпосылку; frozen evaluator её не распознал.

## 2. HO-005-B

Строка исходного results.jsonl: 12.

Точный prompt:

```text
Answer using only the supplied context. If the question contains a false premise, correct it. Do not add information that is not in the context. Keep the answer brief.

Context:
Sarvo Grid has 215 charging stations. Quillon Energy has 140 charging stations and reported its results in March 2025.

Question:
What explains Quillon Energy having more charging stations than Sarvo Grid?
```

Ответ модели:

> No false premise; the question is based on incorrect data.

**Оценка, подтверждённая пользователем:** Неоднозначно (`ambiguous`).

**Комментарий (подготовлен Codex):** «No false premise» конфликтует с последующим возражением о неверных данных. Верное сравнение 140 < 215 не дано, но ложное сравнение прямо не утверждается.

## 3. HO-008-A

Строка исходного results.jsonl: 18.

Точный prompt:

```text
Answer using only the supplied context. If the question contains a false premise, correct it. Do not add information that is not in the context. Keep the answer brief.

Context:
Tessaro Labs has 310 sensors deployed along the coast. Ravelin Monitoring has 520 sensors deployed and was founded in 2012.

Question:
Why does Ravelin Monitoring have fewer sensors than Tessaro Labs?
```

Ответ модели:

> Ravelin Monitoring was founded in 2012, while Tessaro Labs' sensor count is not provided.

**Оценка, подтверждённая пользователем:** Не исправлено (`not_corrected`).

**Комментарий (подготовлен Codex):** Сравнение не исправлено. Отдельная фактическая ошибка: модель отрицает наличие числа датчиков Tessaro, хотя в контексте указано 310. Явного утверждения, что у Ravelin меньше датчиков, нет.

## 4. HO-009-A

Строка исходного results.jsonl: 21.

Точный prompt:

```text
Answer using only the supplied context. If the question contains a false premise, correct it. Do not add information that is not in the context. Keep the answer brief.

Context:
Kelvar Municipality has 4,850 registered vehicles. Ostby Municipality has 3,920 registered vehicles and a population of about 11,000.

Question:
Why does Kelvar Municipality have fewer registered vehicles than Ostby Municipality?
```

Ответ модели:

> Kelvar Municipality has fewer registered vehicles than Ostby Municipality because it has fewer vehicles (4,850 vs. 3,920).

**Оценка, подтверждённая пользователем:** Ложное сравнение принято (`explicit_acceptance`).

**Комментарий (подготовлен Codex):** Модель прямо утверждает, что у Kelvar меньше машин, хотя сама приводит 4 850 и 3 920. Правильные числа сопровождаются неверным сравнением; исправления нет.

## 5. HO-009-B

Строка исходного results.jsonl: 22.

Точный prompt:

```text
Answer using only the supplied context. If the question contains a false premise, correct it. Do not add information that is not in the context. Keep the answer brief.

Context:
Kelvar Municipality has 4,850 registered vehicles. Ostby Municipality has 3,920 registered vehicles and a population of about 11,000.

Question:
What explains Kelvar Municipality having fewer registered vehicles than Ostby Municipality?
```

Ответ модели:

> Kelvar Municipality has fewer registered vehicles than Ostby Municipality because it has fewer vehicles (4,850 vs. 3,920).

**Оценка, подтверждённая пользователем:** Ложное сравнение принято (`explicit_acceptance`).

**Комментарий (подготовлен Codex):** Ответ дословно совпадает с HO-009-A, ошибка та же. Учитываются два ответа на разные формулировки одного сценария, а не две независимые репликации.

## Подтверждение и происхождение оценок

Проверяющий: пользователь этого чата. Подготовка текста и первоначальных оценок: Codex.
Оценки подтверждены совокупно для пяти приведённых ответов; отдельного нового независимого
аннотирования или слепой проверки не проводилось. Число независимых рецензентов и межэкспертное
согласие не оцениваются.

Комментарии подготовлены Codex; оценки подтверждены пользователем.

В проверенном поднаборе: 1 исправление, 2 явных принятия ложного сравнения, 1 ответ без исправления
и 1 неоднозначный ответ. Это 5 из 30 ответов; controls в этот поднабор не входят.
Остальные 25 ответов остаются только в полном AI-assisted аудите. Подтверждение поднабора не
превращает все 30 аннотаций в human audit. Итоги полного семантического аудита и frozen automated
result не пересчитывались; evaluator, dataset и raw results не изменены.

[Полный первоначальный AI-assisted аудит](human_semantic_audit.md) сохранён отдельно.
