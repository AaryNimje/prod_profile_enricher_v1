import os
import json
import requests
import csv
from typing import Dict, List, Any, Optional
import time
from pydantic import BaseModel, Field, field_validator
from datetime import datetime

class InputProfile(BaseModel):
    """Input profile data from CSV/Excel"""
    first_name: str = Field(..., min_length=1, description="Person's first name")
    last_name: str = Field(..., min_length=1, description="Person's last name")
    company: str = Field(..., min_length=1, description="Company name")
    position: str = Field(..., min_length=1, description="Job position/title")
    email: Optional[str] = Field(None, description="Email address if available")
    
    @property
    def full_name(self) -> str:
        """Get full name"""
        return f"{self.first_name} {self.last_name}".strip()
    
    @field_validator('email')
    @classmethod
    def validate_email(cls, v):
        """Validate email format"""
        if v and v.lower() in ['nan', 'null', '', 'none']:
            return None
        return v

class EnrichedProfile(BaseModel):
    """Enriched profile data from Sonar research"""
    Name: str = Field(..., description="Full verified name")
    Company: str = Field(..., description="Current company name")
    Designation: str = Field(..., description="Current job title/position")
    Linkedin_URL: Optional[str] = Field(None, description="LinkedIn profile URL")
    Email: Optional[str] = Field(None, description="Professional email address")
    Phone_Number: Optional[str] = Field(None, description="Phone number")
    Citations: List[str] = Field(..., min_items=1, max_items=10, description="Research sources")
    About: str = Field(..., min_length=10, max_length=500, description="Professional summary")  # Increased limit
    
    @field_validator('Name', 'Company', 'Designation', 'About')
    @classmethod
    def validate_not_null(cls, v):
        """Ensure required fields are not null"""
        if v is None:
            raise ValueError("Field cannot be null")
        return str(v).strip()
    
    @field_validator('About')
    @classmethod
    def validate_about_length(cls, v):
        """Truncate About section if too long"""
        if v and len(v) > 500:
            return v[:497] + "..."
        return v
    
    @field_validator('Citations')
    @classmethod
    def validate_citations_length(cls, v):
        """Ensure we have exactly 10 citations"""
        if len(v) != 10:
            # Pad with empty strings or truncate to 10
            if len(v) < 10:
                v.extend(["Manual research required"] * (10 - len(v)))
            else:
                v = v[:10]
        return v

class APIResponse(BaseModel):
    """API response wrapper"""
    success: bool
    profile: Optional[EnrichedProfile] = None
    error: Optional[str] = None
    processing_time: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=datetime.now)

class EnrichmentResults(BaseModel):
    """Final results container"""
    total_processed: int
    successful: int
    failed: int
    profiles: List[EnrichedProfile]
    processing_time: float
    timestamp: datetime = Field(default_factory=datetime.now)

class ProfileEnricher:
    """Profile enrichment using Sonar deep research API with Pydantic validation"""
    
    def __init__(self, model_name: str = "sonar-deep-research"):
        """Initialize with API key from environment and model selection"""
        self.api_key = os.environ.get('PERPLEXITY_API_KEY')
        if not self.api_key:
            raise ValueError("PERPLEXITY_API_KEY not found in environment variables")
        
        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Available models for deep research
        self.available_models = {
            "sonar-deep-research": {
                "name": "sonar-deep-research",
                "description": "Deep research model for comprehensive profile analysis",
                "max_tokens": 4000,
                "temperature": 0.1
            },
            "sonar-pro": {
                "name": "sonar-pro", 
                "description": "Fast research model with citations",
                "max_tokens": 2000,
                "temperature": 0.2
            },
            "sonar": {
                "name": "sonar",
                "description": "Lightweight search model",
                "max_tokens": 1500,
                "temperature": 0.3
            }
        }
        
        if model_name not in self.available_models:
            print(f"⚠️ Model '{model_name}' not found. Available models:")
            for model_id, info in self.available_models.items():
                print(f"   • {model_id}: {info['description']}")
            model_name = "sonar-deep-research"
            
        self.current_model = self.available_models[model_name]
        print(f"🤖 Using model: {self.current_model['name']} - {self.current_model['description']}")
        
    def create_research_prompt(self, profile: InputProfile) -> str:
        """Create optimized prompt for deep research"""
        return f"""You are a professional researcher conducting deep background research. Find comprehensive information about this person:

PERSON TO RESEARCH:
- Name: {profile.full_name}
- Company: {profile.company}
- Position: {profile.position}

RESEARCH REQUIREMENTS:
1. Find their current professional information
2. Locate their LinkedIn profile URL
3. Search for contact information (email, phone)
4. Gather career background and achievements
5. Look for recent news, articles, or mentions

RESPONSE FORMAT (JSON only):
{{
    "Name": "Full verified name",
    "Company": "Current company name",
    "Designation": "Current job title/position", 
    "Linkedin_URL": "Direct LinkedIn profile URL or null",
    "Email": "Professional email address or null",
    "Phone_Number": "Phone number or null",
    "Citations": ["source1", "source2", "source3", "source4", "source5", "source6", "source7", "source8", "source9", "source10"],
    "About": "50-word professional summary including key achievements, expertise, and current role"
}}

IMPORTANT INSTRUCTIONS:
- Return ONLY valid JSON, no additional text
- Use null for missing information, not "Not found" or empty strings
- Provide exactly 10 citations (URLs or source names)
- About section must be approximately 50 words
- Verify information accuracy before including
- Focus on professional LinkedIn, company websites, and reputable business sources"""

    def enrich_profile(self, input_profile: InputProfile) -> APIResponse:
        """Enrich a single profile using Sonar deep research"""
        
        start_time = time.time()
        
        try:
            prompt = self.create_research_prompt(input_profile)
            
            payload = {
                "model": self.current_model["name"],
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": self.current_model["temperature"],
                "max_tokens": self.current_model["max_tokens"],
                "stream": False
            }
            
            print(f"🔍 Researching: {input_profile.full_name} at {input_profile.company}")
            
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=300  # Increased to 5 minutes for deep research
            )
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                # Parse JSON response
                try:
                    # Clean the response - sometimes API returns extra text
                    content = content.strip()
                    if content.startswith('```json'):
                        content = content[7:]
                    if content.endswith('```'):
                        content = content[:-3]
                    content = content.strip()
                    
                    # Try to extract JSON if there's extra text
                    start_idx = content.find('{')
                    end_idx = content.rfind('}') + 1
                    if start_idx != -1 and end_idx != 0:
                        content = content[start_idx:end_idx]
                    
                    raw_data = json.loads(content)
                    
                    # Handle null values from API
                    if raw_data.get('Name') is None:
                        raw_data['Name'] = input_profile.full_name
                    if raw_data.get('Company') is None:
                        raw_data['Company'] = input_profile.company
                    if raw_data.get('Designation') is None:
                        raw_data['Designation'] = input_profile.position
                    if raw_data.get('About') is None:
                        raw_data['About'] = f"{input_profile.full_name} works as {input_profile.position} at {input_profile.company}."
                    
                    # Add original email if provided and not found
                    if input_profile.email and not raw_data.get('Email'):
                        raw_data['Email'] = input_profile.email
                    
                    # Validate with Pydantic
                    enriched_profile = EnrichedProfile(**raw_data)
                    
                    print(f"✅ Successfully enriched: {enriched_profile.Name}")
                    
                    return APIResponse(
                        success=True,
                        profile=enriched_profile,
                        processing_time=processing_time
                    )
                    
                except json.JSONDecodeError as e:
                    print(f"⚠️ Invalid JSON response for {input_profile.full_name}: {e}")
                    print(f"Raw content preview: {content[:300]}...")
                    fallback_profile = self._create_fallback_profile(input_profile)
                    return APIResponse(
                        success=False,
                        profile=fallback_profile,
                        error=f"JSON decode error: {str(e)}",
                        processing_time=processing_time
                    )
                except Exception as e:
                    print(f"⚠️ Validation error for {input_profile.full_name}: {e}")
                    fallback_profile = self._create_fallback_profile(input_profile)
                    return APIResponse(
                        success=False,
                        profile=fallback_profile,
                        error=f"Validation error: {str(e)}",
                        processing_time=processing_time
                    )
            else:
                error_text = response.text
                print(f"❌ API Error {response.status_code} for {input_profile.full_name}: {error_text}")
                
                # Show available models if invalid model error
                if "invalid model" in error_text.lower() or response.status_code == 400:
                    print("💡 Available models for Perplexity:")
                    for model_id, info in self.available_models.items():
                        print(f"   • {model_id}")
                
                fallback_profile = self._create_fallback_profile(input_profile)
                return APIResponse(
                    success=False,
                    profile=fallback_profile,
                    error=f"API Error {response.status_code}: {error_text}",
                    processing_time=processing_time
                )
                
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"❌ Exception for {input_profile.full_name}: {str(e)}")
            fallback_profile = self._create_fallback_profile(input_profile)
            return APIResponse(
                success=False,
                profile=fallback_profile,
                error=str(e),
                processing_time=processing_time
            )
    
    def _create_fallback_profile(self, input_profile: InputProfile) -> EnrichedProfile:
        """Create fallback profile when API fails"""
        return EnrichedProfile(
            Name=input_profile.full_name,
            Company=input_profile.company,
            Designation=input_profile.position,
            Linkedin_URL=None,
            Email=input_profile.email,
            Phone_Number=None,
            Citations=["API request failed - manual research required"] * 10,
            About=f"{input_profile.full_name} works as {input_profile.position} at {input_profile.company}. Professional background and achievements require manual research due to API limitations."
        )
    
    def process_csv_file(self, file_path: str, delay_seconds: float = 10.0) -> EnrichmentResults:
        """Process CSV file and enrich all profiles"""
        
        start_time = time.time()
        
        print(f"📁 Loading CSV file: {file_path}")
        
        # Read CSV file
        profiles_data = []
        with open(file_path, 'r', encoding='utf-8-sig', newline='') as csvfile:
            # Try to detect delimiter
            sample = csvfile.read(1024)
            csvfile.seek(0)
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter
            
            reader = csv.DictReader(csvfile, delimiter=delimiter)
            headers = reader.fieldnames
            print(f"📋 Detected columns: {headers}")
            
            for row in reader:
                profiles_data.append(row)
        
        print(f"📊 Found {len(profiles_data)} profiles to process")
        
        # Map common column name variations
        column_mapping = self._detect_columns(headers)
        print(f"🎯 Column mapping: {column_mapping}")
        
        # Parse input profiles with Pydantic validation
        input_profiles = []
        for i, row in enumerate(profiles_data):
            try:
                # Extract person data
                first_name = str(row.get(column_mapping.get('first_name', ''), '')).strip()
                last_name = str(row.get(column_mapping.get('last_name', ''), '')).strip()
                company = str(row.get(column_mapping.get('company', ''), '')).strip()
                position = str(row.get(column_mapping.get('position', ''), '')).strip()
                email = str(row.get(column_mapping.get('email', ''), '')).strip()
                
                # Skip empty rows
                if not first_name or not last_name or first_name.lower() in ['nan', ''] or last_name.lower() in ['nan', '']:
                    continue
                
                # Clean email
                if email.lower() in ['nan', '', 'none', 'null']:
                    email = None
                
                # Create and validate input profile
                input_profile = InputProfile(
                    first_name=first_name,
                    last_name=last_name,
                    company=company,
                    position=position,
                    email=email
                )
                
                input_profiles.append(input_profile)
                
            except Exception as e:
                print(f"⚠️ Skipping row {i + 1}: {e}")
                continue
        
        print(f"✅ Validated {len(input_profiles)} profiles for processing")
        print(f"⏱️  Estimated time: {len(input_profiles) * delay_seconds / 60:.1f} minutes")
        
        # Process each profile
        enriched_profiles = []
        successful = 0
        failed = 0
        
        for i, input_profile in enumerate(input_profiles):
            print(f"\n👤 Processing {i + 1}/{len(input_profiles)}: {input_profile.full_name}")
            
            # Enrich profile
            api_response = self.enrich_profile(input_profile)
            
            if api_response.profile:
                enriched_profiles.append(api_response.profile)
                if api_response.success:
                    successful += 1
                    print(f"   ✅ Success in {api_response.processing_time:.1f}s")
                else:
                    failed += 1
                    print(f"   ❌ Failed: {api_response.error}")
            
            # Rate limiting - Deep research needs longer delays
            if i < len(input_profiles) - 1:  # Don't delay after last request
                print(f"⏳ Waiting {delay_seconds} seconds for rate limiting...")
                time.sleep(delay_seconds)
        
        total_time = time.time() - start_time
        
        return EnrichmentResults(
            total_processed=len(input_profiles),
            successful=successful,
            failed=failed,
            profiles=enriched_profiles,
            processing_time=total_time
        )
    
    def _detect_columns(self, columns: List[str]) -> Dict[str, str]:
        """Detect column names automatically"""
        mapping = {}
        
        # First name detection
        for col in columns:
            if any(term in col.lower() for term in ['first', 'fname', 'first_name']):
                mapping['first_name'] = col
                break
        
        # Last name detection
        for col in columns:
            if any(term in col.lower() for term in ['last', 'lname', 'last_name', 'surname']):
                mapping['last_name'] = col
                break
        
        # Email detection
        for col in columns:
            if any(term in col.lower() for term in ['email', 'mail', 'e-mail']):
                mapping['email'] = col
                break
        
        # Company detection
        for col in columns:
            if any(term in col.lower() for term in ['company', 'organization', 'org', 'employer']):
                mapping['company'] = col
                break
        
        # Position detection
        for col in columns:
            if any(term in col.lower() for term in ['position', 'title', 'job', 'role', 'designation']):
                mapping['position'] = col
                break
        
        return mapping
    
    def save_results(self, results: EnrichmentResults, output_file: str = 'enriched_profiles.json'):
        """Save enriched profiles to JSON file"""
        
        # Convert to dict for JSON serialization
        output_data = {
            "metadata": {
                "total_processed": results.total_processed,
                "successful": results.successful,
                "failed": results.failed,
                "processing_time": results.processing_time,
                "timestamp": results.timestamp.isoformat()
            },
            "profiles": [profile.model_dump() for profile in results.profiles]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {output_file}")


def convert_excel_to_csv(excel_path: str) -> str:
    """Convert Excel file to CSV for processing"""
    try:
        import openpyxl
        
        # Load workbook
        wb = openpyxl.load_workbook(excel_path)
        ws = wb.active
        
        # Create CSV filename
        csv_path = excel_path.replace('.xlsx', '.csv').replace('.xls', '.csv')
        
        # Write to CSV
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            for row in ws.iter_rows(values_only=True):
                # Filter out None values and convert to strings
                clean_row = [str(cell) if cell is not None else '' for cell in row]
                writer.writerow(clean_row)
        
        print(f"📄 Converted {excel_path} to {csv_path}")
        return csv_path
        
    except ImportError:
        print("❌ openpyxl not available. Please convert Excel to CSV manually or install openpyxl")
        raise
    except Exception as e:
        print(f"❌ Error converting Excel to CSV: {e}")
        raise


def main():
    """Main function to run profile enrichment"""
    
    # Check for API key
    if not os.environ.get('PERPLEXITY_API_KEY'):
        print("❌ Error: PERPLEXITY_API_KEY environment variable not set")
        print("💡 In PowerShell, set it using: $env:PERPLEXITY_API_KEY='your_api_key_here'")
        print("💡 Or in Command Prompt: set PERPLEXITY_API_KEY=your_api_key_here")
        return
    
    # Show available models
    print("🤖 AVAILABLE MODELS:")
    models = {
        "sonar-deep-research": "🔍 Best for comprehensive research (slower, most detailed)",
        "sonar-pro": "⚡ Balanced speed and depth with citations",  
        "sonar": "🚀 Fastest with basic search"
    }
    
    for model, desc in models.items():
        print(f"   • {model}: {desc}")
    
    # Get model choice
    model_choice = input("\n🎯 Choose model (default: sonar-deep-research): ").strip()
    if not model_choice:
        model_choice = "sonar-deep-research"
    
    # Initialize enricher
    try:
        enricher = ProfileEnricher(model_choice)
    except Exception as e:
        print(f"❌ Failed to initialize ProfileEnricher: {e}")
        return
    
    # Get file path from user
    file_path = input("\n📁 Enter path to CSV/Excel file: ").strip().strip('"')
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    try:
        # Handle Excel files
        if file_path.lower().endswith(('.xlsx', '.xls')):
            try:
                csv_path = convert_excel_to_csv(file_path)
                file_path = csv_path
            except Exception as e:
                print(f"❌ Could not convert Excel file: {e}")
                print("💡 Please convert to CSV manually or install openpyxl")
                return
        
        # Set delay based on model
        delay_map = {
            "sonar-deep-research": 15.0,  # Deep research takes time
            "sonar-pro": 5.0,
            "sonar": 3.0
        }
        delay = delay_map.get(model_choice, 10.0)
        
        # Process CSV file
        results = enricher.process_csv_file(file_path, delay_seconds=delay)
        
        # Display results summary
        print(f"\n🎉 Processing completed!")
        print(f"📊 Total processed: {results.total_processed}")
        print(f"✅ Successful: {results.successful}")
        print(f"❌ Failed: {results.failed}")
        print(f"⏱️  Total time: {results.processing_time/60:.1f} minutes")
        print("\n" + "="*80)
        
        # Display detailed results (first 3 profiles)
        displayed_count = min(3, len(results.profiles))
        for i, profile in enumerate(results.profiles[:displayed_count], 1):
            print(f"\n📋 Profile {i}:")
            print(f"Name: {profile.Name}")
            print(f"Company: {profile.Company}")
            print(f"Designation: {profile.Designation}")
            print(f"LinkedIn: {profile.Linkedin_URL}")
            print(f"Email: {profile.Email}")
            print(f"Phone: {profile.Phone_Number}")
            print(f"About: {profile.About[:100]}{'...' if len(profile.About) > 100 else ''}")
            print(f"Citations: {len(profile.Citations)} sources")
            print("-" * 50)
        
        if len(results.profiles) > displayed_count:
            print(f"... and {len(results.profiles) - displayed_count} more profiles")
        
        # Save results
        save_choice = input("\n💾 Save results to JSON file? (y/n): ").lower()
        if save_choice == 'y':
            output_file = input("📝 Enter output filename (default: enriched_profiles.json): ").strip()
            if not output_file:
                output_file = 'enriched_profiles.json'
            enricher.save_results(results, output_file)
        
        print("\n✅ Profile enrichment completed!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
