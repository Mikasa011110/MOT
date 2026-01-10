# AI2-THOR 常见：FloorPlan1-30 Kitchens, 201-230 Living, 301-330 Bedrooms, 401-430 Bathrooms
# 这里给一个默认 split：每类前 20 train，后 5 test（共 80/20）
TRAIN_SCENES = (
    [f"FloorPlan{i}" for i in range(1, 21)] +
    [f"FloorPlan{200+i}" for i in range(1, 21)] +
    [f"FloorPlan{300+i}" for i in range(1, 21)] +
    [f"FloorPlan{400+i}" for i in range(1, 21)]
)

TEST_SCENES = (
    [f"FloorPlan{i}" for i in range(21, 26)] +
    [f"FloorPlan{200+i}" for i in range(21, 26)] +
    [f"FloorPlan{300+i}" for i in range(21, 26)] +
    [f"FloorPlan{400+i}" for i in range(21, 26)]
)

# paper target sets (keep close to paper) :contentReference[oaicite:2]{index=2}
TARGETS = {
    "Kitchen": ["Toaster", "Microwave", "Fridge", "CoffeeMachine", "GarbageCan", "Bowl"],
    "LivingRoom": ["Pillow", "Laptop", "Television", "GarbageCan", "Bowl"],
    "Bedroom": ["HousePlant", "Lamp", "Book", "AlarmClock"],
    "Bathroom": ["Sink", "ToiletPaper", "SoapBottle", "LightSwitch"],
}
