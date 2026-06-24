import os
import sys
import time
import json
import sqlite3
from datetime import datetime
import numpy as np

# Make sure parent workspace and local test directories are in path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import database_chroma_new as database

# 50 Ground Truth Memories
TEST_MEMORIES = [
    # Schedule & Trainer
    {"id": 1, "tag": "semantic", "subtag": "implicit", "q": "Who is your personal trainer and when do you meet?", "r": "My personal trainer is named Marcus, and we meet every Wednesday at 5 PM."},
    {"id": 2, "tag": "semantic", "subtag": "explicit", "q": "When was your fitness assessment done?", "r": "Had a fitness assessment on June 12, 2026. Measured body fat at 18.5 percent."},
    {"id": 3, "tag": "semantic", "subtag": "implicit", "q": "What is your preference for morning workouts?", "r": "I prefer morning workouts at 6:30 AM before going to the office."},
    {"id": 4, "tag": "procedural", "subtag": "explicit", "q": "What is the dynamic warm-up sequence?", "r": "Always start with 5 minutes of rowing, followed by 10 leg swings, 10 arm circles, and 5 bodyweight squats."},
    {"id": 5, "tag": "semantic", "subtag": "implicit", "q": "Who is your backup trainer?", "r": "If Marcus is busy, my backup trainer is Sarah, who works on Saturday mornings."},
    
    # Injuries & Pain
    {"id": 6, "tag": "semantic", "subtag": "explicit", "q": "Do you have lower back pain?", "r": "I suffer from chronic lower back pain and must avoid heavy deadlifts or bent-over rows."},
    {"id": 7, "tag": "semantic", "subtag": "implicit", "q": "Do you have shoulder issues?", "r": "I experience shoulder impingement on the right side and avoid overhead pressing."},
    {"id": 8, "tag": "semantic", "subtag": "explicit", "q": "When did you hurt your wrist?", "r": "Sprained my left wrist during clean and jerk on May 5, 2026. Avoid wrist-heavy extension."},
    {"id": 9, "tag": "semantic", "subtag": "implicit", "q": "Do you have knee pain?", "r": "My left knee has mild patellar tendonitis, so I limit deep lunges and jumping exercises."},
    {"id": 10, "tag": "semantic", "subtag": "explicit", "q": "Do you have asthma?", "r": "I have mild exercise-induced asthma. Always keep my inhaler in my gym bag."},
    
    # Goals & Targets
    {"id": 11, "tag": "semantic", "subtag": "implicit", "q": "What is your current calorie deficit target?", "r": "My dietary goal is to maintain a daily caloric deficit of 400 calories for fat loss."},
    {"id": 12, "tag": "semantic", "subtag": "explicit", "q": "What is your current weight and target weight?", "r": "My current body weight is 82 kilograms, and my target weight is 78 kilograms."},
    {"id": 13, "tag": "semantic", "subtag": "implicit", "q": "What is your 10k running goal?", "r": "My current running goal is to complete a 10k race in under 50 minutes by October."},
    {"id": 14, "tag": "semantic", "subtag": "explicit", "q": "What is your bench press goal?", "r": "My strength goal is to bench press 100 kilograms for a single rep by the end of the year."},
    {"id": 15, "tag": "semantic", "subtag": "implicit", "q": "What is your body fat percentage goal?", "r": "Targeting a body fat percentage of 12 percent by the end of the summer cutting cycle."},
    
    # Nutrition & Supplementation
    {"id": 16, "tag": "semantic", "subtag": "explicit", "q": "Do you take creatine?", "r": "I take 5 grams of creatine monohydrate daily post-workout with grape juice."},
    {"id": 17, "tag": "semantic", "subtag": "implicit", "q": "What are your dietary preferences?", "r": "My dietary preference is vegetarian high-protein, eating plenty of tofu, tempeh, and red lentils."},
    {"id": 18, "tag": "semantic", "subtag": "explicit", "q": "Do you drink protein shakes?", "r": "Drink a whey protein isolate shake with almond milk and a banana immediately after weightlifting."},
    {"id": 19, "tag": "semantic", "subtag": "implicit", "q": "What is your daily protein target?", "r": "My daily protein intake target is 160 grams to support muscle hypertrophy."},
    {"id": 20, "tag": "semantic", "subtag": "explicit", "q": "Do you take pre-workout?", "r": "I avoid caffeine-based pre-workout because it interferes with my sleep; I use L-citrulline instead."},
    
    # Preferences & Equipment
    {"id": 21, "tag": "semantic", "subtag": "implicit", "q": "Do you prefer barbell or dumbbell bench press?", "r": "I prefer using a barbell for bench press rather than dumbbells or machines."},
    {"id": 22, "tag": "semantic", "subtag": "explicit", "q": "Where do you prefer to run?", "r": "My preferred cardio exercise is trail running in the forest rather than treadmill or road running."},
    {"id": 23, "tag": "semantic", "subtag": "implicit", "q": "What is your favorite barbell brand?", "r": "I prefer training with Rogue barbells because of their superior knurling and spin."},
    {"id": 24, "tag": "semantic", "subtag": "explicit", "q": "What shoes do you use for squatting?", "r": "I wear flat-soled barefoot shoes for squatting to improve foot stability and drive."},
    {"id": 25, "tag": "semantic", "subtag": "implicit", "q": "Do you use lifting straps?", "r": "I only use lifting straps for heavy pull-ups or Romanian deadlifts when grip fatigue is the bottleneck."},
    
    # Routines & Splits
    {"id": 26, "tag": "procedural", "subtag": "explicit", "q": "What is your weekly workout split?", "r": "My current workout split is a 4-day Upper-Lower routine: Upper on Monday/Thursday, Lower on Tuesday/Friday."},
    {"id": 27, "tag": "procedural", "subtag": "implicit", "q": "What is your core routine?", "r": "My core finisher is 3 rounds of: 15 hanging leg raises, 30-second plank, and 12 cable woodchops per side."},
    {"id": 28, "tag": "procedural", "subtag": "explicit", "q": "How do you track your sets?", "r": "I log every working set in a paper notebook, including reps, weight, and rate of perceived exertion (RPE)."},
    {"id": 29, "tag": "procedural", "subtag": "implicit", "q": "What rest times do you use?", "r": "Rest 3 minutes between heavy compound sets, and 90 seconds for isolation or accessory exercises."},
    {"id": 30, "tag": "procedural", "subtag": "explicit", "q": "What is your stretching routine?", "r": "Do static stretching post-workout: focus on hamstrings, hip flexors, and chest wall opening for 30 seconds each."},
    
    # Hydration & Recovery
    {"id": 31, "tag": "semantic", "subtag": "implicit", "q": "What are your hydration targets?", "r": "On workout days, my target water intake is 4 liters; on rest days, it is 3 liters."},
    {"id": 32, "tag": "semantic", "subtag": "explicit", "q": "Do you take epsom salt baths?", "r": "I take an epsom salt bath on Sunday evenings to relieve muscle soreness and relax before the week starts."},
    {"id": 33, "tag": "semantic", "subtag": "implicit", "q": "What recovery tools do you use?", "r": "I use a foam roller on my quads and lats for 10 minutes every night before bed."},
    {"id": 34, "tag": "semantic", "subtag": "explicit", "q": "Do you drink alcohol?", "r": "I avoid drinking alcohol during active training cycles to prevent disruption of protein synthesis and recovery."},
    {"id": 35, "tag": "semantic", "subtag": "implicit", "q": "What is your recovery heart rate?", "r": "My resting heart rate is 52 beats per minute, which indicates solid aerobic fitness."},
    
    # Specific Exercise Details
    {"id": 36, "tag": "procedural", "subtag": "explicit", "q": "How do you do squats?", "r": "Squat to parallel or below, pushing knees outwards, keeping chest up, and driving through the heels."},
    {"id": 37, "tag": "procedural", "subtag": "implicit", "q": "How do you perform bicep curls?", "r": "Keep elbows locked at the sides, avoid swinging the torso, and squeeze the biceps at the top of the range."},
    {"id": 38, "tag": "procedural", "subtag": "explicit", "q": "What is your pull-up protocol?", "r": "Perform strict pull-ups with a shoulder-width pronated grip, focus on driving elbows down to engage the lats."},
    {"id": 39, "tag": "procedural", "subtag": "implicit", "q": "How do you execute lunges?", "r": "Step forward, lowering hips until back knee is near floor, keeping torso upright, and pushing back to start."},
    {"id": 40, "tag": "procedural", "subtag": "explicit", "q": "How do you do lateral raises?", "r": "Raise dumbbells to the sides with a slight elbow bend, lead with the elbows, and stop at shoulder height."},
    
    # Logs & Diary (Episodic)
    {"id": 41, "tag": "semantic", "subtag": "implicit", "q": "How did your bench press session go yesterday?", "r": "Felt strong during yesterday's bench press; completed 4 sets of 8 reps at 80 kilograms."},
    {"id": 42, "tag": "semantic", "subtag": "explicit", "q": "Did you miss any workouts recently?", "r": "Missed the leg workout on June 4, 2026, due to a late meeting at the office."},
    {"id": 43, "tag": "semantic", "subtag": "implicit", "q": "How was your recovery sleep last night?", "r": "Slept poorly last night (only 5.5 hours); felt fatigued and lethargic during my morning training session."},
    {"id": 44, "tag": "semantic", "subtag": "explicit", "q": "What did you eat for breakfast today?", "r": "Had oatmeal with blueberries, chia seeds, and a scoop of vegan vanilla protein powder for breakfast today."},
    {"id": 45, "tag": "semantic", "subtag": "implicit", "q": "How did your trail run go on Saturday?", "r": "Completed a 8-mile forest trail run on Saturday. Heart rate averaged 148 beats per minute."},
    {"id": 46, "tag": "semantic", "subtag": "explicit", "q": "How did your knees feel on Tuesday?", "r": "My left knee was aching slightly after squats on Tuesday. I applied ice for 15 minutes."},
    {"id": 47, "tag": "semantic", "subtag": "implicit", "q": "Did you take a rest day recently?", "r": "Took a complete rest day on Thursday. Did some light walking and foam rolling for recovery."},
    {"id": 48, "tag": "semantic", "subtag": "explicit", "q": "Did you hit a PR recently?", "r": "Hit a personal record of 12 strict pull-ups in a single set during Monday's workout."},
    {"id": 49, "tag": "semantic", "subtag": "implicit", "q": "How was your nutrition over the weekend?", "r": "Fell off my caloric deficit over the weekend due to a friend's birthday dinner. Ate pizza and cake."},
    {"id": 50, "tag": "semantic", "subtag": "explicit", "q": "What did you weigh this morning?", "r": "Weighed in at 81.8 kilograms this morning. The scale is slowly moving downwards."}
]

# Generate 200 Paraphrased Queries (4 per Memory)
EVAL_QUERIES = []
for mem in TEST_MEMORIES:
    mid = mem["id"]
    tag = mem["tag"]
    
    if mid == 1:
        paraphrases = [
            "Who is my coach and when do we meet up?",
            "Details about my training schedule with Marcus",
            "What day do I have a personal training session at 5 PM?",
            "When is my workout appointment with Marcus?"
        ]
    elif mid == 2:
        paraphrases = [
            "What did my fitness evaluation show?",
            "June 12 body fat measurement records",
            "My body fat percent from my latest assessment",
            "When did I measure my body composition?"
        ]
    elif mid == 3:
        paraphrases = [
            "What time do I prefer to work out in the morning?",
            "Do I train before going to work?",
            "My 6:30 AM training preferences",
            "Morning exercise schedule before office hours"
        ]
    elif mid == 4:
        paraphrases = [
            "How should I warm up dynamically?",
            "Sequence for warming up before lift",
            "Should I do rowing or squats for warm up?",
            "WARM UP dynamic sequence exercises"
        ]
    elif mid == 5:
        paraphrases = [
            "Who covers my sessions if Marcus is out?",
            "Sarah Saturday morning training details",
            "My secondary backup personal trainer",
            "If Marcus is unavailable, who coaches me?"
        ]
    elif mid == 6:
        paraphrases = [
            "Is deadlifting safe with lower back issues?",
            "What exercises should I avoid for my back pain?",
            "I have chronic lower back problems, can I do bent over rows?",
            "Workout modifications for chronic lower back injury"
        ]
    elif mid == 7:
        paraphrases = [
            "Can I do military press with right shoulder impingement?",
            "Why do I avoid overhead pressing?",
            "Right side shoulder pain exercise limits",
            "Is overhead pressing okay with shoulder impingement?"
        ]
    elif mid == 8:
        paraphrases = [
            "When did I hurt my left wrist?",
            "Wrist injury during clean and jerk",
            "May 5 wrist sprain details",
            "Wrist extension exercises to avoid due to sprain"
        ]
    elif mid == 9:
        paraphrases = [
            "Left knee tendonitis workout adjustments",
            "Should I avoid lunges and jumping for knee pain?",
            "Patellar tendonitis knee issues",
            "Mild knee pain recovery modifications"
        ]
    elif mid == 10:
        paraphrases = [
            "Where do I keep my inhaler?",
            "Asthma during workouts and safety steps",
            "What if I have an asthma attack during training?",
            "Gym bag inhaler reminder for asthma"
        ]
    elif mid == 11:
        paraphrases = [
            "What is my target calorie deficit?",
            "How many calories deficit for fat loss?",
            "What calorie deficit daily goal do I have?",
            "Fat loss daily calorie reduction target"
        ]
    elif mid == 12:
        paraphrases = [
            "What is my goal weight in kg?",
            "Current weight vs target weight in kilograms",
            "How much do I weigh now and what is my target?",
            "I want to weigh 78 kg, what is my current weight?"
        ]
    elif mid == 13:
        paraphrases = [
            "What is my target time for the 10k run?",
            "October 10k running race goal",
            "Running goal: under 50 minutes details",
            "10k race target speed and date"
        ]
    elif mid == 14:
        paraphrases = [
            "My single rep bench press max strength goal",
            "Can I bench press 100 kg by end of year?",
            "Bench press target kilograms",
            "100kg bench press goal details"
        ]
    elif mid == 15:
        paraphrases = [
            "Target body fat percent for summer cut",
            "Summer cutting cycle bodyfat goal",
            "What body fat percentage am I aiming for?",
            "12 percent body fat target details"
        ]
    elif mid == 16:
        paraphrases = [
            "How do I take my daily creatine?",
            "Do I take creatine post-workout or pre-workout?",
            "Creatine with grape juice recipe",
            "5g creatine monohydrate supplementation timing"
        ]
    elif mid == 17:
        paraphrases = [
            "Am I vegetarian and what are my protein sources?",
            "Dietary preference high protein tofu lentil",
            "Do I eat meat or follow a plant-based diet?",
            "Vegetarian bodybuilding diet sources"
        ]
    elif mid == 18:
        paraphrases = [
            "What do I drink immediately after lifting?",
            "Whey protein isolate shake with banana almond milk",
            "Post weightlifting recovery protein shake recipe",
            "When do I drink a whey protein shake?"
        ]
    elif mid == 19:
        paraphrases = [
            "How much protein do I need per day?",
            "My daily target for protein intake in grams",
            "Hypertrophy protein goal 160g",
            "Daily protein target for muscle building"
        ]
    elif mid == 20:
        paraphrases = [
            "Why do I avoid caffeine pre-workouts?",
            "L-citrulline instead of standard preworkout",
            "What do I take instead of caffeine preworkout?",
            "Sleep issues due to caffeine pre-workout"
        ]
    elif mid == 21:
        paraphrases = [
            "Do I prefer barbell or dumbbells for benching?",
            "Bench press barbell vs machine preference",
            "Which equipment do I like for flat bench press?",
            "Barbell bench press preference detail"
        ]
    elif mid == 22:
        paraphrases = [
            "Do I like trail running or road running?",
            "Forest running preference details",
            "Preferred outdoor cardio trail forest",
            "Do I run on a treadmill or in nature?"
        ]
    elif mid == 23:
        paraphrases = [
            "Which brand of barbells do I prefer?",
            "Rogue barbells knurling spin preference",
            "Favorite gym barbell manufacturer",
            "Why do I like training with Rogue bar?"
        ]
    elif mid == 24:
        paraphrases = [
            "What shoes do I wear for squats?",
            "Flat-soled barefoot shoes for lifting",
            "Squat footwear preference",
            "Do I wear running shoes or flat shoes for squatting?"
        ]
    elif mid == 25:
        paraphrases = [
            "When do I use grip lifting straps?",
            "Romanian deadlift grip straps utility",
            "Do I use straps for heavy pull-ups?",
            "Lifting straps for grip fatigue bottlenecks"
        ]
    elif mid == 26:
        paraphrases = [
            "What is my current 4-day workout routine split?",
            "Upper-Lower routine days of the week",
            "Do I train lower body on Tuesdays?",
            "Workout split Monday Tuesday Thursday Friday"
        ]
    elif mid == 27:
        paraphrases = [
            "What core exercises do I do as a finisher?",
            "Hanging leg raises plank cable woodchops core routine",
            "My ab training finisher sequence",
            "3 rounds ab finisher details"
        ]
    elif mid == 28:
        paraphrases = [
            "How do I track workout progress?",
            "Do I use a paper notebook or app for logs?",
            "Logging reps, weight, RPE in notebook",
            "Workout tracking paper log methodology"
        ]
    elif mid == 29:
        paraphrases = [
            "How long do I rest between heavy compound lifts?",
            "Rest period between isolation exercise sets",
            "Is 90 seconds rest enough for isolation movements?",
            "Rest interval target for heavy strength lifts"
        ]
    elif mid == 30:
        paraphrases = [
            "What is my post-workout stretching split?",
            "Focus areas for static stretching recovery",
            "Hamstring hip flexor chest static stretch duration",
            "Should I stretch post exercise?"
        ]
    elif mid == 31:
        paraphrases = [
            "What is my water target on workout days vs rest days?",
            "How many liters of water do I drink on days off?",
            "Workout day hydration targets in liters",
            "Rest day water target vs exercise day"
        ]
    elif mid == 32:
        paraphrases = [
            "Do I take epsom salt baths for recovery?",
            "Sunday evening muscle soreness relief routine",
            "Why do I take epsom salt bath on Sundays?",
            "Sunday relaxation muscle bath soak"
        ]
    elif mid == 33:
        paraphrases = [
            "What recovery tool do I use on quads and lats?",
            "Foam rolling routine before bed",
            "How long do I foam roll at night?",
            "Daily recovery habits foam rolling"
        ]
    elif mid == 34:
        paraphrases = [
            "Do I drink beer or liquor during active cycles?",
            "Alcohol impact on protein synthesis and recovery",
            "Why do I avoid drinking alcohol?",
            "Active training cycle recovery habits and alcohol"
        ]
    elif mid == 35:
        paraphrases = [
            "What is my resting heart rate?",
            "52 bpm resting pulse rate details",
            "My cardiovascular conditioning heart rate",
            "Is my resting heart rate high or low?"
        ]
    elif mid == 36:
        paraphrases = [
            "How do I squat properly?",
            "Knee and chest placement during squats",
            "Squatting depth parallel heel drive",
            "What is the form checklist for back squats?"
        ]
    elif mid == 37:
        paraphrases = [
            "Form rules for strict bicep curls",
            "Should I swing my torso during curls?",
            "Bicep contraction elbows locked form guide",
            "Proper arm curl technique"
        ]
    elif mid == 38:
        paraphrases = [
            "How do I perform a correct strict pull-up?",
            "Shoulder-width grip lat engagement pullups",
            "Driving elbows down during vertical pulling",
            "Pronated grip pullups technique"
        ]
    elif mid == 39:
        paraphrases = [
            "Proper form guidelines for lunges",
            "Back knee depth and torso positioning lunges",
            "How do I lowering hips in walking lunges?",
            "Torso alignment during forward lunges"
        ]
    elif mid == 40:
        paraphrases = [
            "How do I do side shoulder raises?",
            "Dumbbell lateral raise elbow angle and height",
            "Lead with elbows lateral raise shoulder form",
            "Lateral raise height limit"
        ]
    elif mid == 41:
        paraphrases = [
            "Yesterday's bench press sets and reps log",
            "How much weight did I bench yesterday?",
            "Bench press 80kg 4 sets 8 reps check",
            "Did I feel strong on bench press yesterday?"
        ]
    elif mid == 42:
        paraphrases = [
            "Why did I miss a leg workout on June 4?",
            "Late office meeting workout cancellation",
            "June 4 missed lower body exercise details",
            "Missed workout leg day reason"
        ]
    elif mid == 43:
        paraphrases = [
            "Poor sleep impact on recovery last night",
            "Why was I fatigued during morning training?",
            "5.5 hours sleep session log",
            "Did I sleep well yesterday night?"
        ]
    elif mid == 44:
        paraphrases = [
            "What did I eat for breakfast this morning?",
            "Oatmeal blueberries chia seed protein breakfast",
            "My vegan vanilla breakfast bowl recipe",
            "Morning meal ingredients log"
        ]
    elif mid == 45:
        paraphrases = [
            "Saturday trail running length and pulse rate",
            "8-mile forest trail run log check",
            "148 bpm average heart rate Saturday run",
            "Aerobic run details on Saturday"
        ]
    elif mid == 46:
        paraphrases = [
            "Did my knee hurt on Tuesday?",
            "Post squat knee pain icing session",
            "Tuesday left knee ache duration",
            "Squats left knee pain recovery steps"
        ]
    elif mid == 47:
        paraphrases = [
            "When did I take a complete recovery rest day?",
            "Thursday rest day activities log",
            "Did I walk or foam roll on my day off?",
            "Thursday active recovery diary"
        ]
    elif mid == 48:
        paraphrases = [
            "Did I hit a strict pull-up PR recently?",
            "Monday pullups 12 reps personal record",
            "My latest strict pull-ups milestone log",
            "How many pull-ups did I do in Monday's set?"
        ]
    elif mid == 49:
        paraphrases = [
            "Birthday dinner calorie surplus slip weekend",
            "Pizza and cake cheat meal weekend",
            "Nutrition slip-up diary weekend details",
            "Did I follow my caloric deficit over the weekend?"
        ]
    elif mid == 50:
        paraphrases = [
            "What was my weigh-in weight this morning?",
            "Morning scale weight log 81.8 kilograms",
            "Is my weight slowly dropping check",
            "Today's morning bodyweight record"
        ]
        
    for text in paraphrases:
        EVAL_QUERIES.append({
            "target_id": mid,
            "tag": tag,
            "query": text
        })

# 50 Unrelated Distractor Queries (Should return empty due to threshold)
DISTRACTOR_QUERIES = [
    "What is the capital city of France?",
    "How does a internal combustion engine work?",
    "Who painted the Mona Lisa?",
    "Explain quantum computing in simple terms",
    "How do you cook spaghetti carbonara?",
    "What is the distance from the Earth to the Moon?",
    "Tell me a funny joke about programming",
    "What are the three laws of thermodynamics?",
    "How do plants perform photosynthesis?",
    "What is the plot of Shakespeare's Hamlet?",
    "How do you write a binary search algorithm in Python?",
    "Who was the first president of the United States?",
    "What is the difference between SQL and NoSQL?",
    "How do bees make honey?",
    "What causes a volcanic eruption?",
    "How do you tie a double windsor knot?",
    "What are the benefits of learning a second language?",
    "Who wrote the book 1984?",
    "How does GPS technology pinpoint location?",
    "What is the chemical symbol for gold?",
    "How are rainbow colors formed in the sky?",
    "Explain the rules of cricket",
    "What is the history of the Great Wall of China?",
    "How do you create a clean git merge request?",
    "What are the symptoms of water damage in a laptop?",
    "Who discovered penicillin?",
    "What is the definition of inflation in economics?",
    "How do you change a flat tire on a car?",
    "What is the lifecycle of a butterfly?",
    "How do you play chess opening principles?",
    "What is the origin of Halloween?",
    "How do solar panels convert sunlight to electricity?",
    "What is the difference between TCP and UDP?",
    "Who invented the telephone?",
    "What are the main functions of the central bank?",
    "How do you calculate compound interest?",
    "What is the deep learning transformer architecture?",
    "How do you start a campfire safely?",
    "Who is the author of Harry Potter?",
    "What is the largest ocean on Earth?",
    "How do you write a resume for software engineering?",
    "What are the rules of soccer?",
    "How do airplanes stay in the air?",
    "What is the difference between JVM, JRE, and JDK?",
    "How do you clean a cast iron skillet?",
    "What is the history of the internet?",
    "Explain the concept of neural networks",
    "What are the benefits of reading daily?",
    "How do you bake sourdough bread?",
    "What is the capital of Japan?",
    "What is the capital of Australia?",
    "How do you calculate the area of a circle?",
    "Who wrote the play Romeo and Juliet?",
    "What is the speed of light in a vacuum?",
    "How does a microwave cook food?",
    "What is the capital of Canada?",
    "Explain the theory of general relativity.",
    "What are the primary colors of light?",
    "How do you make a cup of filter coffee?",
    "What is the difference between compiler and interpreter?",
    "How do tides work in the ocean?",
    "Who was the first man to walk on the moon?",
    "What is the function of red blood cells?",
    "How do you write a hello world program in Rust?",
    "What is the capital of Germany?",
    "Explain the concept of supply and demand.",
    "Who painted the Starry Night?",
    "What is the tallest mountain in the world?",
    "How do you prune a tomato plant?",
    "What is the chemical formula for water?"
]

def run_evaluation():
    print("=============================================================")
    print("STARTING COGNITIVE VECTOR SEARCH EVALUATION FRAMEWORK")
    print("=============================================================")
    print(f"Stored Ground-Truth Memories: {len(TEST_MEMORIES)}")
    print(f"Paraphrased Related Target Queries: {len(EVAL_QUERIES)}")
    print(f"Unrelated Distractor Queries: {len(DISTRACTOR_QUERIES)}")
    print(f"Total Test Cases: {len(EVAL_QUERIES) + len(DISTRACTOR_QUERIES)}")
    print("=============================================================\n")
    
    test_user = "eval_athlete_v2"
    
    # Clean up any previous eval data in database
    conn = sqlite3.connect(database.DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM memories WHERE username = ?", (test_user,))
    conn.commit()
    conn.close()
    
    # 1. Insert Ground-Truth Memories
    print("1. Populating evaluation database with ground-truth records...")
    start_populate = time.time()
    
    # Keep track of generated database IDs
    db_id_map = {} # Maps target_id -> inserted SQLite ID
    
    from datetime import datetime, timedelta
    now = datetime.now()
    
    for idx, mem in enumerate(TEST_MEMORIES):
        hours_ago = (idx * 15) % 720
        timestamp = (now - timedelta(hours=hours_ago)).isoformat()
        
        database.save_memory(
            username=test_user,
            tag=mem["tag"],
            query=mem["q"],
            response=mem["r"],
            subtag=mem["subtag"],
            timestamp=timestamp
        )
        
        # Fetch the auto-incremented ID
        conn = sqlite3.connect(database.DB_PATH)
        c = conn.cursor()
        c.execute("SELECT max(id) FROM memories WHERE username = ?", (test_user,))
        inserted_id = c.fetchone()[0]
        conn.close()
        db_id_map[mem["id"]] = inserted_id
        
        if (idx + 1) % 10 == 0 or (idx + 1) == len(TEST_MEMORIES):
            print(f"   - Stored {idx+1}/{len(TEST_MEMORIES)} memories...")
            
    populate_time = time.time() - start_populate
    print(f" -> Database populated in {populate_time:.2f} seconds.\n")
    
    # 2. Run Related Paraphrases Evaluation
    print("2. Evaluating accuracy and latency of paraphrased queries...")
    latencies = []
    
    top_1_hits = 0
    top_3_hits = 0
    top_5_hits = 0
    misses = 0
    threshold_failures = 0
    
    query_logs = []
    
    for idx, q_case in enumerate(EVAL_QUERIES):
        target_id = q_case["target_id"]
        tag = q_case["tag"]
        query_text = q_case["query"]
        expected_db_id = db_id_map[target_id]
        
        start_q = time.time()
        results = database.vector_query_memories(test_user, tag, query_text, top_k=5)
        duration = (time.time() - start_q) * 1000.0 # ms
        latencies.append(duration)
        
        # Check rank
        retrieved_ids = [r["id"] for r in results]
        
        is_top_1 = False
        is_top_3 = False
        is_top_5 = False
        
        if expected_db_id in retrieved_ids:
            rank = retrieved_ids.index(expected_db_id) + 1
            if rank == 1:
                top_1_hits += 1
                is_top_1 = True
            if rank <= 3:
                top_3_hits += 1
                is_top_3 = True
            if rank <= 5:
                top_5_hits += 1
                is_top_5 = True
        else:
            if not results:
                threshold_failures += 1
            misses += 1
            rank = -1 # Not found
            
        query_logs.append({
            "query": query_text,
            "target_id": target_id,
            "expected_db_id": expected_db_id,
            "retrieved_ids": retrieved_ids,
            "rank": rank,
            "latency_ms": duration
        })
        
        if (idx + 1) % 40 == 0 or (idx + 1) == len(EVAL_QUERIES):
            print(f"   - Evaluated {idx+1}/{len(EVAL_QUERIES)} queries...")
            
    # 3. Run Distractor Queries Evaluation
    print("\n3. Evaluating threshold safety on unrelated distractor queries...")
    distractor_latencies = []
    false_positives = 0
    
    for idx, dist_q in enumerate(DISTRACTOR_QUERIES):
        start_q = time.time()
        # Test across all tags
        results_sem = database.vector_query_memories(test_user, "semantic", dist_q, top_k=1)
        results_epi = database.vector_query_memories(test_user, "episodic", dist_q, top_k=1)
        results_pro = database.vector_query_memories(test_user, "procedural", dist_q, top_k=1)
        duration = (time.time() - start_q) * 1000.0
        distractor_latencies.append(duration)
        
        # If any result returned (escaped the >= 0.35 filter) it's a false positive
        if results_sem or results_epi or results_pro:
            false_positives += 1
            
        if (idx + 1) % 15 == 0 or (idx + 1) == len(DISTRACTOR_QUERIES):
            print(f"   - Checked {idx+1}/{len(DISTRACTOR_QUERIES)} distractors...")
            
    print("\n=============================================================")
    print("EVALUATION METRICS & RESULTS SUMMARY")
    print("=============================================================")
    
    total_rel = len(EVAL_QUERIES)
    acc_top1 = (top_1_hits / total_rel) * 100.0
    acc_top3 = (top_3_hits / total_rel) * 100.0
    acc_top5 = (top_5_hits / total_rel) * 100.0
    
    total_dist = len(DISTRACTOR_QUERIES)
    fpr = (false_positives / total_dist) * 100.0
    
    all_latencies = latencies + distractor_latencies
    avg_lat = np.mean(all_latencies)
    median_lat = np.median(all_latencies)
    p95_lat = np.percentile(all_latencies, 95)
    p99_lat = np.percentile(all_latencies, 99)
    
    print(f"Top-1 Accuracy:  {acc_top1:.2f}% ({top_1_hits}/{total_rel})")
    print(f"Top-3 Accuracy:  {acc_top3:.2f}% ({top_3_hits}/{total_rel})")
    print(f"Top-5 Accuracy:  {acc_top5:.2f}% ({top_5_hits}/{total_rel})")
    print(f"Threshold Misses: {threshold_failures}/{total_rel} (Queries fell below 0.35 relevance threshold)")
    print(f"False Positives:  {false_positives}/{total_dist} (FPR: {fpr:.2f}% - unrelated queries returning results)")
    print("-------------------------------------------------------------")
    print(f"Average Latency: {avg_lat:.2f} ms")
    print(f"Median Latency:  {median_lat:.2f} ms")
    print(f"95th Percentile: {p95_lat:.2f} ms")
    print(f"99th Percentile: {p99_lat:.2f} ms")
    print("=============================================================\n")
    
    # Write JSON results
    summary_results = {
        "metrics": {
            "top_1_accuracy": acc_top1,
            "top_3_accuracy": acc_top3,
            "top_5_accuracy": acc_top5,
            "threshold_misses": threshold_failures,
            "false_positives": false_positives,
            "false_positive_rate": fpr,
            "avg_latency_ms": avg_lat,
            "median_latency_ms": median_lat,
            "p95_latency_ms": p95_lat,
            "p99_latency_ms": p99_lat
        },
        "query_details": query_logs
    }
    
    with open(os.path.join(current_dir, "eval_results.json"), "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=4)
        
    print(f"Saved detailed query logs to '{os.path.join(current_dir, 'eval_results.json')}'")
    
    # 4. Generate the Notebook File programmatically
    generate_notebook(db_id_map, summary_results)

def generate_notebook(db_id_map, summary_results):
    notebook_path = os.path.join(current_dir, "chromadb_search_test.ipynb")
    print(f"Generating Jupyter Notebook: '{notebook_path}'...")
    
    # Prepare text representation of the results for pre-execution display in notebook
    metrics = summary_results["metrics"]
    metrics_str = f"""EVALUATION RESULTS PRE-RUN CACHE SUMMARY:
- Top-1 Retrieval Accuracy: {metrics['top_1_accuracy']:.2f}%
- Top-3 Retrieval Accuracy: {metrics['top_3_accuracy']:.2f}%
- Top-5 Retrieval Accuracy: {metrics['top_5_accuracy']:.2f}%
- False Positive Rate: {metrics['false_positive_rate']:.2f}%
- Average Query Latency: {metrics['avg_latency_ms']:.2f} ms
- p95 Query Latency: {metrics['p95_latency_ms']:.2f} ms"""

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🧠 AI Workout Coach - ChromaDB Semantic Search Evaluation\n",
                "\n",
                "This notebook evaluates the **hybrid retrieval search** (ChromaDB EphemeralClient + SQLite BLOBs + Google Gemini Embeddings + ChromaBm25EmbeddingFunction) and **Cross-Encoder reranking** system used in the AI Workout Coach database storage.\n",
                "\n",
                "### 📈 Retrieval Methodology\n",
                "1. **Semantic Search (ChromaDB EphemeralClient)**: Uses the `GoogleGeminiEmbeddingFunction` (model `gemini-embedding-001`) or falls back to local SentenceTransformer (`nomic-ai/nomic-embed-text-v1.5`) in-memory. Computes Cosine Similarity.\n",
                "2. **Lexical Search (ChromaBm25EmbeddingFunction)**: Native BM25 keyword matching fallback.\n",
                "3. **Hybrid Score Fusion**: Blends scores with a weighted ratio of **70% Semantic + 30% Keyword**.\n",
                "4. **Threshold Filtering**: Only candidates with a combined hybrid score `>= 0.48` or high-confidence semantic matches `>= 0.60` are retained.\n",
                "5. **Cross-Encoder Reranking**: The top 15 candidates are reranked using the `cross-encoder/ms-marco-MiniLM-L-6-v2` model (70% weight) combined with a metadata boost (30% weight) consisting of explicit subtag preferences (60%) and exponential recency decay (40%).\n",
                "\n",
                "### 🔬 Evaluation Dataset Structure\n",
                "- **50 Stored Memories**: Ground-truth user facts covering weekly schedules, body-composition assessments, injuries, dietary targets, and recovery habits.\n",
                "- **200 Paraphrased Queries**: 4 diverse variations per memory (semantic paraphrasing, keyword emphasis, short queries) to evaluate query robustness.\n",
                "- **50 Distractor Queries**: Unrelated general knowledge questions (capital cities, economics, coding, etc.) to verify threshold filter reliability (FPR)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Initialize environments and import modules\n",
                "import os\n",
                "import sys\n",
                "import time\n",
                "import sqlite3\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "# Add parent workspace directory to sys.path to resolve imports correctly\n",
                "notebook_dir = os.getcwd()\n",
                "parent_dir = os.path.abspath(os.path.join(notebook_dir, '..'))\n",
                "if parent_dir not in sys.path:\n",
                "    sys.path.insert(0, parent_dir)\n",
                "if notebook_dir not in sys.path:\n",
                "    sys.path.insert(0, notebook_dir)\n",
                "\n",                "import database_chroma_new as database\n",
                "print(\"Embedding Model status:\", database.get_db_status())"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2. Define the evaluation script data\n",
                "# Storing references to memories and paraphrases\n",
                "from test_semantic_search import TEST_MEMORIES, EVAL_QUERIES, DISTRACTOR_QUERIES\n",
                "print(f\"Loaded {len(TEST_MEMORIES)} ground-truth memories, {len(EVAL_QUERIES)} paraphrases, and {len(DISTRACTOR_QUERIES)} distractors.\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 3. Populate test database and index vectors\n",
                "test_user = \"eval_athlete_notebook\"\n",
                "\n",
                "# Clear out existing test athlete rows\n",
                "conn = sqlite3.connect(database.DB_PATH)\n",
                "c = conn.cursor()\n",
                "c.execute(\"DELETE FROM memories WHERE username = ?\", (test_user,))\n",
                "conn.commit()\n",
                "conn.close()\n",
                "\n",
                "print(\"Populating memories...\")\n",
                "db_id_map = {}\n",
                "from datetime import datetime, timedelta\n",
                "now = datetime.now()\n",
                "for idx, mem in enumerate(TEST_MEMORIES):\n",
                "    hours_ago = (idx * 15) % 720\n",
                "    timestamp = (now - timedelta(hours=hours_ago)).isoformat()\n",
                "    database.save_memory(\n",
                "        username=test_user,\n",
                "        tag=mem[\"tag\"],\n",
                "        query=mem[\"q\"],\n",
                "        response=mem[\"r\"],\n",
                "        subtag=mem[\"subtag\"],\n",
                "        timestamp=timestamp\n",
                "    )\n",
                "    \n",
                "    conn = sqlite3.connect(database.DB_PATH)\n",
                "    c = conn.cursor()\n",
                "    c.execute(\"SELECT max(id) FROM memories WHERE username = ?\", (test_user,))\n",
                "    db_id_map[mem[\"id\"]] = c.fetchone()[0]\n",
                "    conn.close()\n",
                "\n",
                "print(f\"Populated evaluation database with {len(db_id_map)} records.\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 4. Run retrieval evaluation loop\n",
                "print(\"Running paraphrased query tests...\")\n",
                "latencies = []\n",
                "ranks = []\n",
                "threshold_misses = 0\n",
                "\n",
                "for idx, q_case in enumerate(EVAL_QUERIES):\n",
                "    target_id = q_case[\"target_id\"]\n",
                "    tag = q_case[\"tag\"]\n",
                "    query_text = q_case[\"query\"]\n",
                "    expected_id = db_id_map[target_id]\n",
                "    \n",
                "    t_start = time.time()\n",
                "    results = database.vector_query_memories(test_user, tag, query_text, top_k=5)\n",
                "    latencies.append((time.time() - t_start) * 1000.0)\n",
                "    \n",
                "    retrieved_ids = [r[\"id\"] for r in results]\n",
                "    if expected_id in retrieved_ids:\n",
                "        ranks.append(retrieved_ids.index(expected_id) + 1)\n",
                "    else:\n",
                "        ranks.append(-1)\n",
                "        if not results:\n",
                "            threshold_misses += 1\n",
                "            \n",
                "print(\"Running distractor query tests...\")\n",
                "distractor_latencies = []\n",
                "false_positives = 0\n",
                "for dist_q in DISTRACTOR_QUERIES:\n",
                "    t_start = time.time()\n",
                "    res_sem = database.vector_query_memories(test_user, \"semantic\", dist_q, top_k=1)\n",
                "    res_epi = database.vector_query_memories(test_user, \"episodic\", dist_q, top_k=1)\n",
                "    res_pro = database.vector_query_memories(test_user, \"procedural\", dist_q, top_k=1)\n",
                "    distractor_latencies.append((time.time() - t_start) * 1000.0)\n",
                "    if res_sem or res_epi or res_pro:\n",
                "        false_positives += 1\n",
                "\n",
                "print(\"Evaluation loop complete.\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 5. Calculate and display metrics\n",
                "total_queries = len(EVAL_QUERIES)\n",
                "top_1 = sum(1 for r in ranks if r == 1)\n",
                "top_3 = sum(1 for r in ranks if 1 <= r <= 3)\n",
                "top_5 = sum(1 for r in ranks if 1 <= r <= 5)\n",
                "\n",
                "acc_top1 = (top_1 / total_queries) * 100\n",
                "acc_top3 = (top_3 / total_queries) * 100\n",
                "acc_top5 = (top_5 / total_queries) * 100\n",
                "fpr = (false_positives / len(DISTRACTOR_QUERIES)) * 100\n",
                "\n",
                "all_latencies = latencies + distractor_latencies\n",
                "\n",
                "print(\"===============================================\")\n",
                "print(\"SEMANTIC VECTOR SEARCH RETRIEVAL PERFORMANCE\")\n",
                "print(\"===============================================\")\n",
                "print(f\"Top-1 Accuracy:  {acc_top1:.2f}% ({top_1}/{total_queries})\")\n",
                "print(f\"Top-3 Accuracy:  {acc_top3:.2f}% ({top_3}/{total_queries})\")\n",
                "print(f\"Top-5 Accuracy:  {acc_top5:.2f}% ({top_5}/{total_queries})\")\n",
                "print(f\"Threshold Misses: {threshold_misses}/{total_queries}\")\n",
                "print(f\"False Positives:  {false_positives}/{len(DISTRACTOR_QUERIES)} (FPR: {fpr:.2f}%)\")\n",
                "print(\"-----------------------------------------------\")\n",
                "print(f\"Average Latency: {np.mean(all_latencies):.2f} ms\")\n",
                "print(f\"Median Latency:  {np.median(all_latencies):.2f} ms\")\n",
                "print(f\"95th Percentile: {np.percentile(all_latencies, 95):.2f} ms\")\n",
                "print(f\"99th Percentile: {np.percentile(all_latencies, 99):.2f} ms\")\n",
                "print(\"===============================================\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 6. Plot performance distributions\n",
                "sns.set_theme(style=\"whitegrid\")\n",
                "plt.figure(figsize=(12, 5))\n",
                "\n",
                "# Latency Distribution\n",
                "plt.subplot(1, 2, 1)\n",
                "sns.histplot(all_latencies, kde=True, bins=30, color=\"skyblue\")\n",
                "plt.axvline(np.mean(all_latencies), color=\"red\", linestyle=\"--\", label=f\"Mean: {np.mean(all_latencies):.1f}ms\")\n",
                "plt.axvline(np.percentile(all_latencies, 95), color=\"orange\", linestyle=\"-.\", label=f\"p95: {np.percentile(all_latencies, 95):.1f}ms\")\n",
                "plt.title(\"Query Latency Distribution (ms)\")\n",
                "plt.xlabel(\"Latency (ms)\")\n",
                "plt.ylabel(\"Count\")\n",
                "plt.legend()\n",
                "\n",
                "# Retrieval Accuracy by Rank\n",
                "plt.subplot(1, 2, 2)\n",
                "categories = ['Top-1', 'Top-3', 'Top-5', 'Miss/Filter']\n",
                "counts = [\n",
                "    top_1,\n",
                "    top_3 - top_1,\n",
                "    top_5 - top_3,\n",
                "    total_queries - top_5\n",
                "]\n",
                "colors = [\"#2ecc71\", \"#3498db\", \"#9b59b6\", \"#e74c3c\"]\n",
                "plt.bar(categories, counts, color=colors)\n",
                "plt.title(\"Retrieval Accuracy by Rank Placement\")\n",
                "plt.xlabel(\"Retrieved Rank\")\n",
                "plt.ylabel(\"Query Count\")\n",
                "for i, val in enumerate(counts):\n",
                "    pct = (val / total_queries) * 100\n",
                "    plt.text(i, val + 2, f\"{pct:.1f}%\", ha='center', fontweight='bold')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        }

    ]

    # Add interactive custom query cell for the user
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 7. Test your own custom queries here!\n",
            "my_query = \"Who is my coach and when do we meet up?\"\n",
            "my_tag = \"semantic\"  # Options: 'semantic', 'episodic', 'procedural'\n",
            "\n",
            "results = database.vector_query_memories(test_user, my_tag, my_query, top_k=3)\n",
            "\n",
            "print(f\"=== SEARCH RESULTS FOR: '{my_query}' (Tag: {my_tag}) ===\\n\")\n",
            "if not results:\n",
            "    print(\"No matches found above the threshold.\")\n",
            "for idx, r in enumerate(results):\n",
            "    print(f\"[{idx+1}] Match (Subtag: {r['subtag']})\")\n",
            "    print(f\"    Prompt:   {r['query']}\")\n",
            "    print(f\"    Memory:   {r['response']}\")\n",
            "    print(f\"    Recorded: {r['timestamp']}\\n\")"
        ]
    })

    # Add diagnostic score breakdown cells for explicit/implicit & recency testing
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 8. Run a detailed diagnostic search on custom queries (Score breakdown)\n",
            "from query_test import run_debug_search\n",
            "\n",
            "my_query = \"Am I vegetarian and what are my protein sources?\"\n",
            "my_tag = \"semantic\"\n",
            "\n",
            "run_debug_search(test_user, my_tag, my_query)"
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 9. Test explicit subtag preference and recency boost behavior\n",
            "from query_test import run_debug_search\n",
            "\n",
            "my_query = \"I will have healthy food only\"\n",
            "my_tag = \"semantic\"\n",
            "\n",
            "run_debug_search(test_user, my_tag, my_query)"
        ]
    })

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=4)
    print(f" -> Notebook successfully generated at '{notebook_path}'.")

if __name__ == "__main__":
    run_evaluation()
