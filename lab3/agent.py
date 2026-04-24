import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

load_dotenv()

# --- 1. Инструменты (Skills) ---
@tool
def calculate_bmr(weight: float, height: float, age: int, gender: str) -> str:
    """Рассчитывает базовый обмен веществ (BMR) по формуле Миффлина-Сан Жеора."""
    gender = gender.lower()
    if gender not in ("male", "female"):
        return "Укажите пол строго как 'male' или 'female'."
    bmr = 10 * weight + 6.25 * height - 5 * age + (5 if gender == "male" else -161)
    daily_calories = bmr * 1.55
    return f"BMR: {bmr:.0f} ккал. Норма для поддержания: ~{daily_calories:.0f} ккал/день."

@tool
def get_exercises_for_muscle(muscle_group: str) -> str:
    """Возвращает список упражнений для указанной мышечной группы."""
    db = {
        "руки": "1. Подъем гантелей на бицепс: 4×10-12\n2. Французский жим: 3×12\n3. Молотки: 3×10",
        "грудь": "1. Жим штанги лежа: 4×8-10\n2. Жим гантелей на наклонной: 4×10-12\n3. Разводка: 3×15",
        "спина": "1. Подтягивания: 4×макс\n2. Тяга штанги в наклоне: 4×10\n3. Тяга верхнего блока: 3×12",
        "ноги": "1. Приседания со штангой: 4×8-10\n2. Жим ногами: 4×12\n3. Румынская тяга: 3×10",
        "плечи": "1. Жим гантелей сидя: 4×10\n2. Махи в стороны: 4×12\n3. Тяга к подбородку: 3×12"
    }
    return db.get(muscle_group.lower(), f"Группа '{muscle_group}' не найдена. Доступно: {', '.join(db.keys())}")

tools = [calculate_bmr, get_exercises_for_muscle]

# --- 2. LLM (LM Studio) ---
llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "qwen/qwen3-4b-2507"),
    base_url=os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1"),
    api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
    temperature=0.1,
    max_tokens=1024,
)

# --- 3. Системный промпт ---
SYSTEM_PROMPT = """Ты — профессиональный фитнес-тренер.
Составляй программы тренировок, подбирай упражнения, давай рекомендации по питанию.
Всегда используй инструменты перед ответом.
Если запрос НЕ связан с фитнесом/тренировками/питанием спортсменов — отвечай строго: "Команда неизвестна".
Отвечай на русском, структурированно, без воды."""

agent = create_react_agent(
    llm,
    tools,
    prompt=SYSTEM_PROMPT,  # ✅ строка вместо state_modifier
)

def ask(question: str) -> str:
    """Локальный тест агента."""
    response = agent.invoke({"messages": [HumanMessage(content=question)]})
    return response["messages"][-1].content

if __name__ == "__main__":
    print("\n🏋️ Тест: программа на руки")
    print(ask("Распиши мне программу тренировок на тренировку мышц рук."))
    print("\n🍎 Тест: расчет калорий")
    print(ask("Мужчина, 25 лет, 80 кг, 180 см. Сколько калорий нужно?"))
    print("\n❌ Тест: неизвестная команда")
    print(ask("Напиши рецепт борща"))