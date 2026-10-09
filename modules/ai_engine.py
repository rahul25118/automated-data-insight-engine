import os
import json
import time
from google import genai

# प्राथमिकता सूची: पहला बिजी हो तो तुरंत दूसरे पर स्विच होगा
ACTIVE_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-3.8-flash"
]

def call_gemini(client, prompt: str) -> str:
    """503 या हाई ट्रैफिक होने पर अपने-आप अगले उपलब्ध मॉडल पर स्विच करता है।"""
    last_err = None
    for model_name in ACTIVE_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            if response and response.text:
                return response.text
        except Exception as e:
            err_str = str(e)
            last_err = err_str
            # अगर 503 (हाई डिमांड) या 429 (रेट लिमिट) या 404 है, तो 1 सेकंड रुककर अगला मॉडल आज़माएँ
            time.sleep(1)
            continue
    raise Exception(f"सभी मॉडल्स पर कॉल विफल रही। अंतिम एरर: {last_err}")

def generate_data_narrative(df_summary: dict, api_key: str) -> str:
    """डेटा समरी मेट्रिक्स के आधार पर Gemini से बिजनेस नैरेटिव इनसाइट्स जनरेट करता है।"""
    clean_key = api_key.strip() if api_key else ""
    if not clean_key:
        return "⚠️ कृपया साइडबार में मान्य Gemini API Key दर्ज करें।"

    try:
        client = genai.Client(api_key=clean_key)
        prompt = f"""
आप एक सीनियर बिजनेस डेटा एनालिस्ट हैं। नीचे दिए गए डेटासेट की समरी और स्टैटिस्टिकल मेट्रिक्स को ध्यान से देखें और एक पेशेवर बिजनेस रिपोर्ट तैयार करें:

डेटा मेट्रिक्स:
- कुल पंक्तियाँ (Rows): {df_summary.get('rows')}
- कुल कॉलम (Columns): {df_summary.get('columns')}
- मिसिंग सेल्स (%): {df_summary.get('missing_percent')}%
- डुप्लिकेट पंक्तियाँ: {df_summary.get('duplicate_rows')}
- न्यूमेरिकल कॉलम: {df_summary.get('numerical_cols')}
- कैटेगोरिकल कॉलम: {df_summary.get('categorical_cols')}
- डेटा कॉलम सूची: {df_summary.get('columns_list')}

कृपया 3 स्पष्ट अनुभागों में संक्षिप्त, बिजनेस-ओरिएंटेड विश्लेषण दें (Markdown फॉर्मेट में):
1. 📌 **एग्जीक्यूटिव समरी (Executive Summary)**: डेटा की समग्र गुणवत्ता और ढाँचा।
2. ⚠️ **संभावित जोखिम और डेटा हाइजीन (Data Hygiene & Risks)**: मिसिंग वैल्यूज, डुप्लिकेट्स या क्लीनिंग की जरूरत।
3. 💡 **व्यावसायिक अनुशंसाएं (Actionable Recommendations)**: इस डेटा से बिजनेस टीम्स को क्या विश्लेषण या कदम उठाने चाहिए।
"""
        return call_gemini(client, prompt)
    except Exception as e:
        return f"❌ AI इनसाइट्स जनरेट करने में त्रुटि: {str(e)}"

def query_data_with_llm(user_question: str, df_schema_info: str, api_key: str) -> dict:
    """यूज़र के सवाल को सुरक्षित Pandas कोड में बदलता है।"""
    clean_key = api_key.strip() if api_key else ""
    if not clean_key:
        return {"error": "⚠️ कृपया साइडबार में मान्य Gemini API Key दर्ज करें।"}

    try:
        client = genai.Client(api_key=clean_key)
        prompt = f"""
आप एक एक्सपर्ट पाइथन डेटा एनालिस्ट हैं। 
डेटाफ्रेम का नाम `df` है। 

डेटाफ्रेम स्कीमा और सैंपल डेटा:
{df_schema_info}

यूज़र का सवाल:
"{user_question}"

निर्देश:
1. यूज़र के सवाल का जवाब देने के लिए शुद्ध Python/Pandas कोड लिखें।
2. कोड का परिणाम हमेशा एक चर `result` में स्टोर होना चाहिए (उदा. result = df.groupby(...)... या result = df[...].head())।
3. केवल df और pd का उपयोग करें। कोई सिस्टम कमांड्स (os, sys) न लिखें।
4. अपना उत्तर केवल वैध JSON प्रारूप में दें:
{{"code": "pandas code here", "explanation": "explanation here"}}
"""
        raw_text = call_gemini(client, prompt).strip()
        
        start_idx = raw_text.find('{')
        end_idx = raw_text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            clean_json = raw_text[start_idx:end_idx+1]
            return json.loads(clean_json)
        return {"error": "AI रिस्पॉन्स से JSON नहीं मिला।"}
    except Exception as e:
        return {"error": f"क्वेरी जनरेट करने में त्रुटि: {str(e)}"}
