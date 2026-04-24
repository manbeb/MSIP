import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool

# Загрузка переменных окружения
load_dotenv()


# --- 1. Инструменты (Skills) ---
@tool
def calculate_bmr(weight: float, height: float, age: int, gender: str) -> str:
    """Рассчитывает базовый обмен веществ (BMR) и суточную норму калорий для планирования питания.
    Аргументы: вес (кг), рост (см), возраст (лет), пол ('male' или 'female')."""
    gender = gender.lower()
    if gender not in ("male", "female"):
        return "Укажите пол строго как 'male' или 'female'."

    bmr = 10 * weight + 6.25 * height - 5 * age + (5 if gender == "male" else -161)
    # Коэффициент умеренной активности (тренировки 3-5 раз в неделю)
    daily_calories = bmr * 1.55
    return f"BMR: {bmr:.0f} ккал. Рекомендуемая норма для поддержания формы: ~{daily_calories:.0f} ккал/день."


@tool
def get_exercises_for_muscle(muscle_group: str) -> str:
    """Возвращает список упражнений для указанной мышечной группы с подходами и повторениями."""
    db = {
        "руки": "1. Подъем гантелей на бицепс: 4x10-12\n2. Французский жим: 3x12\n3. Молотки: 3x10\n4. Разгибания рук на блоке: 4x12",
        "грудь": "1. Жим штанги лежа: 4x8-10\n2. Жим гантелей на наклонной скамье: 4x10-12\n3. Разводка гантелей: 3x15\n4. Отжимания от пола: 3xмакс",
        "спина": "1. Подтягивания широким хватом: 4xмакс\n2. Тяга штанги в наклоне: 4x10\n3. Тяга верхнего блока к груди: 3x12\n4. Гиперэкстензия: 3x15",
        "ноги": "1. Приседания со штангой: 4x8-10\n2. Жим ногами в тренажере: 4x12\n3. Румынская тяга: 3x10\n4. Подъемы на носки стоя: 4x15",
        "плечи": "1. Жим гантелей сидя: 4x10\n2. Махи гантелями в стороны: 4x12\n3. Тяга к подбородку: 3x12\n4. Обратные разведения в тренажере: 3x15"
    }
    key = muscle_group.lower()
    return db.get(key,
                  f"Упражнения для '{muscle_group}' не найдены. Доступные группы: руки, грудь, спина, ноги, плечи.")


tools = [calculate_bmr, get_exercises_for_muscle]

# --- 2. Настройка LLM (LM Studio) ---
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen/qwen3-4b-2507")
API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")  # LM Studio игнорирует ключ, но LangChain требует его

llm = ChatOpenAI(
    model=MODEL_NAME,
    base_url=LM_STUDIO_URL,
    api_key=API_KEY,
    temperature=0.1,
    max_tokens=1024,
)

# --- 3. Системный промпт и Агент ---
SYSTEM_PROMPT = """Ты — профессиональный фитнес-тренер и спортивный нутрициолог.
Твоя задача: составлять программы тренировок, подбирать упражнения и давать рекомендации по питанию/восстановлению.
Всегда используй инструменты для расчетов или поиска упражнений перед формированием ответа.
ОТВЕЧАЙ ТОЛЬКО на вопросы, связанные с фитнесом, тренировками, питанием спортсменов или восстановлением.
Если запрос не относится к спорту или фитнесу, ответь строго одной фразой: "Команда неизвестна".
Отвечай четко, структурированно, на русском языке. Не выдумывай данные."""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)


def ask(question: str) -> str:
    """Функция для локального тестирования агента."""
    return agent_executor.invoke({"input": question})["output"]


if __name__ == "__main__":
    print("\n🏋️ Тест агента:")
    print(ask("Распиши мне программу тренировок на тренировку мышц рук."))
    print("\n🍎 Тест инструмента калорий:")
    print(ask("Мужчина, 25 лет, 80 кг, 180 см. Сколько калорий нужно?"))
    print("\n❌ Тест неизвестной команды:")
    print(ask("Напиши рецепт борща"))