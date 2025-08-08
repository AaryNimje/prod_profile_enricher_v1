import os
import json
import requests

def test_single_api_call():
    """Test a single API call to see what we're getting back"""
    
    api_key = os.environ.get('PERPLEXITY_API_KEY')
    if not api_key:
        print("❌ Please set PERPLEXITY_API_KEY environment variable")
        return
    
    # Test with one person from your file
    test_prompt = """You are a professional researcher conducting deep background research. Find comprehensive information about this person:

PERSON TO RESEARCH:
- Name: Petah Green
- Company: Google  
- Position: GTM Practice Leader Google Cloud EMEA

RESEARCH REQUIREMENTS:
1. Find their current professional information
2. Locate their LinkedIn profile URL
3. Search for contact information (email, phone)
4. Gather career background and achievements
5. Look for recent news, articles, or mentions

RESPONSE FORMAT (JSON only):
{
    "Name": "Full verified name",
    "Company": "Current company name",
    "Designation": "Current job title/position", 
    "Linkedin_URL": "Direct LinkedIn profile URL or null",
    "Email": "Professional email address or null",
    "Phone_Number": "Phone number or null",
    "Citations": ["source1", "source2", "source3", "source4", "source5", "source6", "source7", "source8", "source9", "source10"],
    "About": "50-word professional summary including key achievements, expertise, and current role"
}

IMPORTANT INSTRUCTIONS:
- Return ONLY valid JSON, no additional text
- Use null for missing information, not "Not found" or empty strings
- Provide exactly 10 citations (URLs or source names)
- About section must be approximately 50 words
- Verify information accuracy before including
- Focus on professional LinkedIn, company websites, and reputable business sources"""

    payload = {
        "model": "sonar-deep-research",
        "messages": [
            {
                "role": "user",
                "content": test_prompt
            }
        ],
        "temperature": 0.1,
        "max_tokens": 4000,
        "stream": False
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    print("🔍 Testing API call...")
    print("⏳ This may take up to 5 minutes for deep research...")
    
    try:
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=payload,
            timeout=300
        )
        
        print(f"\n📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            print(f"\n📄 Raw Response Length: {len(content)} characters")
            print(f"📄 Raw Response Preview:")
            print("-" * 50)
            print(content[:500])
            print("-" * 50)
            
            # Try to clean and parse JSON
            print(f"\n🧹 Cleaning JSON...")
            cleaned_content = content.strip()
            if cleaned_content.startswith('```json'):
                cleaned_content = cleaned_content[7:]
            if cleaned_content.endswith('```'):
                cleaned_content = cleaned_content[:-3]
            cleaned_content = cleaned_content.strip()
            
            # Try to extract JSON if there's extra text
            start_idx = cleaned_content.find('{')
            end_idx = cleaned_content.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_content = cleaned_content[start_idx:end_idx]
            else:
                json_content = cleaned_content
            
            print(f"📄 Cleaned JSON:")
            print("-" * 50)
            print(json_content[:500])
            print("-" * 50)
            
            # Try to parse JSON
            try:
                parsed_data = json.loads(json_content)
                print(f"\n✅ JSON parsed successfully!")
                print(f"📋 Parsed fields:")
                for key, value in parsed_data.items():
                    if isinstance(value, list):
                        print(f"   {key}: [{len(value)} items]")
                    elif isinstance(value, str) and len(value) > 50:
                        print(f"   {key}: {value[:50]}...")
                    else:
                        print(f"   {key}: {value}")
                        
                # Check for issues
                print(f"\n🔍 Issue Analysis:")
                if parsed_data.get('About') and len(parsed_data['About']) > 300:
                    print(f"   ⚠️  About field too long: {len(parsed_data['About'])} chars (max 300)")
                
                null_fields = [k for k, v in parsed_data.items() if v is None and k in ['Name', 'Company', 'Designation', 'About']]
                if null_fields:
                    print(f"   ⚠️  Null required fields: {null_fields}")
                
                if len(parsed_data.get('Citations', [])) != 10:
                    print(f"   ⚠️  Citations count: {len(parsed_data.get('Citations', []))} (expected 10)")
                
                print(f"\n🎯 This is what your code will receive and try to validate!")
                
            except json.JSONDecodeError as e:
                print(f"\n❌ JSON parsing failed: {e}")
                print(f"📄 Failed content: {json_content}")
                
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"📄 Error Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Request Exception: {e}")

if __name__ == "__main__":
    test_single_api_call()