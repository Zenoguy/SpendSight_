# regex_engine/category_rules.py

CATEGORY_REGEX = {
    "Fuel": [
        r"petrol", r"diesel", r"fuel",
        r"hpcl", r"bpcl", r"indian oil"
    ],
    "Dining": [
        r"swiggy", r"zomato", r"restaurant",
        r"cafe", r"eatery", r"food court"
    ],
    "Transport.Cab": [
        r"uber", r"ola", r"rapido", r"cab", r"taxi"
    ],
    "Groceries": [
        r"dmart", r"bigbasket", r"grofers", r"grocery"
    ],
    "Shopping.Online": [
        r"amazon", r"flipkart", r"myntra", r"ajio"
    ],
    "Bills.Utilities": [
        r"electricity bill", r"water bill",
        r"postpaid", r"prepaid recharge"
    ],
    "Fees.Charges": [
        r"charge", r"fee", r"penalty", r"gst"
    ],
    "Income.Salary": [
        r"salary", r"payroll", r"credited by employer"
    ],
    "Transfer.Self": [
        r"to self", r"self transfer", r"own account"
    ]
}
