import os
import json
import requests
import re

def find_linkedin_aggressive(name: str, company: str = "", position: str = ""):
    """Aggressively find LinkedIn URL for a person"""
    
    api_key = os.environ.get('PERPLEXITY_API_KEY')
    if not api_key:
        print("❌ Please set PERPLEXITY_API_KEY environment variable")
        return None
    
    # Create ultra-focused LinkedIn prompt
    search_info = f"Name: {name}"
    if company:
        search_info += f" Company: {company}"
    if position:
        search_info += f" Position: {position}"
    
    prompt = f"""FIND LINKEDIN URL ONLY.

PERSON: {search_info}

SEARCH COMMANDS:
1. "{name} LinkedIn"
2. "{name} {company} LinkedIn" 
3. "{name} {position} LinkedIn"
4. "{name} site:linkedin.com"
5. "{name} professional profile"

RETURN ONLY:
LinkedIn URL: [exact URL]
Email: [email if found]
Phone: [phone if found]

RULES:
- MUST find LinkedIn URL
- Search multiple name variations
- Check company employee pages
- Look in professional directories
- NO extra text, just the information"""

    payload = {
        "model": "sonar-deep-research",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 1000,
        "stream": False
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    print(f"🔍 Aggressively searching for {name}'s LinkedIn...")
    
    try:
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=payload,
            timeout=300
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            print(f"📄 Raw response:")
            print(content)
            print("-" * 50)
            
            # Extract LinkedIn URL using regex
            linkedin_patterns = [
                r'https?://(?:www\.)?linkedin\.com/in/[a-zA-Z0-9\-]+/?',
                r'linkedin\.com/in/[a-zA-Z0-9\-]+/?',
                r'LinkedIn URL:\s*(https?://[^\s]+)',
                r'LinkedIn:\s*(https?://[^\s]+)'
            ]
            
            linkedin_url = None
            for pattern in linkedin_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    linkedin_url = matches[0]
                    if not linkedin_url.startswith('http'):
                        linkedin_url = 'https://' + linkedin_url
                    break
            
            # Extract email
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            email_matches = re.findall(email_pattern, content)
            email = email_matches[0] if email_matches else None
            
            # Extract phone
            phone_patterns = [
                r'\+\d{1,4}[\s\-\(\)]*\d{1,4}[\s\-\(\)]*\d{1,4}[\s\-\(\)]*\d{1,9}',
                r'\(\d{3}\)[\s\-]*\d{3}[\s\-]*\d{4}'
            ]
            phone = None
            for pattern in phone_patterns:
                matches = re.findall(pattern, content)
                if matches:
                    phone = matches[0]
                    break
            
            print(f"🔗 LinkedIn: {linkedin_url or 'NOT FOUND'}")
            print(f"📧 Email: {email or 'NOT FOUND'}")
            print(f"📞 Phone: {phone or 'NOT FOUND'}")
            
            return {
                'linkedin': linkedin_url,
                'email': email,
                'phone': phone,
                'raw_response': content
            }
            
        else:
            print(f"❌ API Error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_linkedin_finder():
    """Test the LinkedIn finder with your people"""
    
    test_people = [
        {"name": "Barton Townley", "company": "Osmii", "position": "Software Development Recruiter"},
        {"name": "Petah Green", "company": "Google", "position": "GTM Practice Leader"},
        {"name": "Shobhit Tiwari", "company": "Publicis Sapient", "position": "Sr. Associate Cloud DevOps"}
    ]
    
    print("🚀 LINKEDIN URL AGGRESSIVE FINDER")
    print("="*50)
    
    for person in test_people:
        print(f"\n👤 Testing: {person['name']}")
        result = find_linkedin_aggressive(
            person['name'], 
            person['company'], 
            person['position']
        )
        
        if result and result['linkedin']:
            print(f"✅ SUCCESS! Found LinkedIn for {person['name']}")
        else:
            print(f"❌ FAILED to find LinkedIn for {person['name']}")
        
        print("-" * 50)

if __name__ == "__main__":
    test_linkedin_finder()