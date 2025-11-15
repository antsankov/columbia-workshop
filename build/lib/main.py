"""
311 Complaint Classifier Exercise
Students: Fill in the PROMPT_TEMPLATE below to classify complaints correctly!
"""

import os
from openai import OpenAI
from collections import defaultdict

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ============================================================================
# STUDENT TASK: Fill in this prompt template to classify the complaints
# ============================================================================
PROMPT_TEMPLATE = """
HI THIS IS A COMPLAINT: {complaint}

Tell me which department should handle this complaint, return one word answer: fire, police, buildings, finance, or parks.
"""
# ============================================================================

# 30 VERY tricky 311 complaints with ground truth labels (multilingual & ambiguous)
COMPLAINTS = [
    # Fire Department - more subtle/ambiguous
    ("Someone's burning trash in a barrel behind the restaurant, not sure if legal", "fire"),
    ("El detector de humo no deja de sonar pero no veo fuego en ningún lado", "fire"),  # Spanish: smoke detector won't stop beeping but no fire anywhere
    ("My landlord removed the sprinkler system during renovations last month", "fire"),
    ("煤气味道很重，可能是邻居的炉子漏气了", "fire"),  # Chinese: Strong gas smell, possibly neighbor's stove leaking
    ("Fire escape being used as storage by tenants, completely blocked", "fire"),
    ("Someone welding in basement without ventilation, sparks everywhere", "fire"),
    
    # Police - overlapping with other departments
    ("Homeless person sleeping in park, harassing visitors for money", "police"),
    ("El vendedor ambulante no tiene permiso y está bloqueando la entrada del metro", "police"),  # Spanish: Street vendor without permit blocking subway entrance
    ("Neighbor's dog bit my kid, owner refuses to provide vaccine records", "police"),
    ("有人在商店门口非法赌博，很多现金交易", "police"),  # Chinese: Illegal gambling at storefront, lots of cash transactions
    ("Car alarm going off every night at 3am for two weeks straight", "police"),
    ("Landlord changed my locks while I was at work, won't let me in", "police"),
    
    # Buildings Department - could be confused with fire/parks
    ("Balcony railing completely rusted through, looks ready to collapse", "buildings"),
    ("La pared del edificio tiene grietas grandes y caen pedazos a la acera", "buildings"),  # Spanish: Building wall has large cracks and pieces falling to sidewalk
    ("Landlord refuses to fix heat in winter, apartment is 45 degrees", "buildings"),
    ("电梯坏了三个月了，老人只能爬楼梯", "buildings"),  # Chinese: Elevator broken for 3 months, elderly must climb stairs
    ("Construction crew working without permits, removed load-bearing wall", "buildings"),
    ("Mold covering entire bathroom ceiling after pipe burst, landlord ignoring", "buildings"),
    
    # Finance/Tax Department - subtle financial issues
    ("Restaurant isn't giving receipts and I think they're dodging taxes", "finance"),
    ("Mi casero no me da recibos de renta para mis impuestos", "finance"),  # Spanish: Landlord won't give rent receipts for taxes
    ("Business closed but still charging my credit card monthly", "finance"),
    ("我的房产税账单有两份，金额不一样", "finance"),  # Chinese: I have two property tax bills with different amounts
    ("Parking meter ate my money, no ticket issued, city won't refund", "finance"),
    ("Contractor demanding cash only, no invoice, suspicious pricing", "finance"),
    
    # Parks Department - overlapping with buildings/police
    ("Tree roots destroying sidewalk, multiple people tripped and fell", "parks"),
    ("El baño del parque está cerrado pero dice que debería estar abierto", "parks"),  # Spanish: Park bathroom is closed but sign says should be open
    ("Rats everywhere in park near overflowing garbage cans", "parks"),
    ("公园的秋千坏了，有锋利的边缘可能伤到孩子", "parks"),  # Chinese: Park swing broken with sharp edges that could hurt children
    ("Someone dumped construction debris on nature trail", "parks"),
    ("Community garden plot being used to sell vegetables illegally", "parks"),
]

def classify_complaint(complaint_text):
    """
    Call OpenAI API to classify a single complaint
    """
    try:
        prompt = PROMPT_TEMPLATE.format(complaint=complaint_text)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=10
        )
        
        prediction = response.choices[0].message.content.strip().lower()
        
        # Validate prediction
        valid_departments = ["fire", "police", "buildings", "finance", "parks"]
        if prediction not in valid_departments:
            return prediction
        
        return prediction
    
    except Exception as e:
        print(f"Error classifying complaint: {e}")
        return "error"

def calculate_metrics(true_labels, predicted_labels):
    """
    Calculate accuracy, precision, and recall for each class
    """
    departments = ["fire", "police", "buildings", "finance", "parks"]
    
    # Overall accuracy
    correct = sum(1 for t, p in zip(true_labels, predicted_labels) if t == p)
    accuracy = correct / len(true_labels) if true_labels else 0
    
    # Per-class metrics
    metrics = {}
    for dept in departments:
        tp = sum(1 for t, p in zip(true_labels, predicted_labels) if t == dept and p == dept)
        fp = sum(1 for t, p in zip(true_labels, predicted_labels) if t != dept and p == dept)
        fn = sum(1 for t, p in zip(true_labels, predicted_labels) if t == dept and p != dept)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[dept] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(1 for t in true_labels if t == dept)
        }
    
    return accuracy, metrics

def main():
    print("=" * 70)
    print("311 COMPLAINT CLASSIFIER - STUDENT EXERCISE")
    print("=" * 70)
    print("\nClassifying 30 complaints across 5 departments...")
    print("This may take a minute...\n")
    
    true_labels = []
    predicted_labels = []
    
    # Classify each complaint
    for i, (complaint, true_dept) in enumerate(COMPLAINTS, 1):
        predicted_dept = classify_complaint(complaint)
        true_labels.append(true_dept)
        predicted_labels.append(predicted_dept)
        
        status = "✅" if predicted_dept == true_dept else "❌"
        print(f"{status} [{i}/30] True: {true_dept:10} | Predicted: {predicted_dept:10}")
        print(f"   '{complaint[:60]}...'")
    
    # Calculate and display metrics
    accuracy, metrics = calculate_metrics(true_labels, predicted_labels)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nOverall Accuracy: {accuracy:.1%} ({sum(1 for t, p in zip(true_labels, predicted_labels) if t == p)}/30)")
    print("\nPer-Department Metrics:")
    print("-" * 70)
    print(f"{'Department':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<8}")
    print("-" * 70)
    
    for dept in ["fire", "police", "buildings", "finance", "parks"]:
        m = metrics[dept]
        print(f"{dept:<12} {m['precision']:>10.1%}  {m['recall']:>10.1%}  {m['f1']:>10.1%}  {m['support']:>6}")
    
    print("-" * 70)
    
    # Provide feedback
    print("\nFEEDBACK:")
    if accuracy >= 0.9:
        print("🎉 Excellent! Your prompt achieves high accuracy.")
    elif accuracy >= 0.75:
        print("👍 Good work! Try to improve edge cases for higher accuracy.")
    elif accuracy >= 0.5:
        print("📚 Your prompt needs refinement. Consider:")
        print("   - Being more specific about what each department handles")
        print("   - Providing clearer distinction between similar categories")
    else:
        print("💡 Keep working on your prompt! Tips:")
        print("   - Define each department's responsibilities clearly")
        print("   - Give examples of complaint types for each department")
        print("   - Use structured output instructions")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: Please set OPENAI_API_KEY environment variable")
        print("In Repl.it: Use the Secrets tab (lock icon) to add OPENAI_API_KEY")
        exit(1)
    
    main()
