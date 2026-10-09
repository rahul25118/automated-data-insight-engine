import os
import json
import re
from openai import OpenAI

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"

def get_client(api_key: str):
    clean_key = api_key.strip() if api_key else ""
    return OpenAI(api_key=clean_key, base_url=DEEPSEEK_BASE_URL)

def generate_data_narrative(df_summary: dict, api_key: str) -> str:
    """डेटा समरी मेट्रिक्स के आधार पर DeepSeek से बिजनेस नैरेटिव इनसाइट्स जनरेट करता है।"""
    if not api_key or not api_key.strip():
        return "⚠️ कृपया साइडबार में मान्य DeepSeek API Key दर्ज करें।"

    try:
        client = get_client(api_key)
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
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a professional enterprise data analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ DeepSeek AI इनसाइट्स जनरेट करने में त्रुटि: {str(e)}"

def query_data_with_llm(user_question: str, df_schema_info: str, api_key: str) -> dict:
    """यूज़र के सवाल को DeepSeek के ज़रिए सटीक Pandas कोड में बदलता है।"""
    if not api_key or not api_key.strip():
        return {"error": "⚠️ कृपया साइडबार में मान्य DeepSeek API Key दर्ज करें।"}

    try:
        client = get_client(api_key)
        prompt = f"""
आप एक एक्सपर्ट Python डेटा एनालिस्ट हैं।
डेटाफ्रेम का नाम `df` है।

डेटाफ्रेम स्कीमा और सैंपल डेटा:
{df_schema_info}

यूज़र का सवाल:
"{user_question}"

निर्देश:
1. यूज़र के सवाल का जवाब देने के लिए शुद्ध Python/Pandas कोड लिखें।
2. कोड का परिणाम हमेशा एक चर `result` में स्टोर होना चाहिए (उदा. result = df.groupby(...)... या result = df[...].head())।
3. केवल df और pd का उपयोग करें। कोई सिस्टम कमांड्स (os, sys, subprocess) न लिखें।
4. अपना उत्तर केवल वैध JSON प्रारूप में दें:
{{"code": "pandas code here", "explanation": "explanation here"}}
"""
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a specialized code generation engine. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        raw_text = response.choices[0].message.content.strip()
        return json.loads(raw_text)
    except Exception as e:
        return {"error": f"DeepSeek क्वेरी में त्रुटि: {str(e)}"}
