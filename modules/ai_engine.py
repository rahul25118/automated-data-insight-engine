import os
from google import genai

def generate_data_narrative(df_summary: dict, api_key: str) -> str:
    """
    डेटा समरी मेट्रिक्स के आधार पर Gemini से बिजनेस नैरेटिव इनसाइट्स जनरेट करता है।
    """
    if not api_key or not api_key.strip():
        return "⚠️ कृपया साइडबार में मान्य Gemini API Key दर्ज करें।"

    try:
        client = genai.Client(api_key=api_key.strip())

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

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"❌ AI इनसाइट्स जनरेट करने में त्रुटि: {str(e)}"
