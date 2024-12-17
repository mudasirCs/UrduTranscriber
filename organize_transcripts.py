import os
from pathlib import Path
import re

# Define category-specific naming templates
CATEGORY_NAMING = {
    "Sunnah": "Sunnah Part {} (Episode {})",
    "Pardah": "Pardah Part {} (Episode {})",
    "Return_of_Jesus": "Return of Jesus Part {} (Episode {})",
    "Night_Journey": "Night Journey Part {} (Episode {})",
    "First_Revelation": "First Revelation Part {} (Episode {})",
    "Heaven_and_Hoor": "Heaven and Hoor Part {} (Episode {})",
    "Music_and_Singing": "Music and Singing Part {} (Episode {})",
    "Meezan_and_Furqan": "Meezan and Furqan Part {} (Episode {})",
    "Usul_e_fiqh": "Usul e Fiqh Part {} (Episode {})",
}

def extract_part_number(filename):
    """Extract part number from filename."""
    # Try to find part number in different formats
    patterns = [
        r'Part[_\s-]+(\d+)',
        r'Series\s+(\d+)',
        r'Episode\s+(\d+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None

def extract_episode_number(filename):
    """Extract episode number (if different from part number)."""
    match = re.search(r'Part[_\s-]+(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def get_date_from_filename(filename):
    """Extract date from filename if it exists."""
    match = re.search(r'(\d{8})_', filename)
    if match:
        return match.group(1)
    return None

def rename_files(base_dir):
    """Rename files according to new convention."""
    base_path = Path(base_dir)
    
    # Process each category directory
    for category_dir in base_path.iterdir():
        if not category_dir.is_dir():
            continue
            
        category_name = category_dir.name
        print(f"\nProcessing {category_name}...")
        
        # Get all docx files in the category
        files = list(category_dir.glob('*.docx'))
        
        # Sort files by part number to handle duplicates
        files.sort(key=lambda x: (extract_part_number(x.name) or 0, x.name))
        
        # Track processed part numbers to handle duplicates
        processed_parts = set()
        
        for file_path in files:
            try:
                if file_path.name.startswith('~$'):  # Skip temporary files
                    continue
                    
                part_num = extract_part_number(file_path.name)
                if not part_num:
                    continue
                
                # Get episode number if different from part number
                episode_num = extract_episode_number(file_path.name) or part_num
                
                # Get date if available
                date_str = get_date_from_filename(file_path.name)
                
                # Generate new name based on category
                if category_name in CATEGORY_NAMING:
                    new_name = CATEGORY_NAMING[category_name].format(
                        len(processed_parts) + 1,  # Sequential part number
                        episode_num
                    )
                else:
                    new_name = f"{category_name} Part {len(processed_parts) + 1} (Episode {episode_num})"
                
                # Add date if available
                if date_str:
                    new_name = f"{new_name} - {date_str}"
                
                # Add extension
                new_name = f"{new_name}.docx"
                
                # Create new path
                new_path = file_path.parent / new_name
                
                # Rename file
                if new_path != file_path:
                    file_path.rename(new_path)
                    print(f"Renamed: {file_path.name} -> {new_name}")
                
                processed_parts.add(part_num)
                
            except Exception as e:
                print(f"Error processing {file_path.name}: {str(e)}")

def main():
    base_dir = "c:/youtube_transcriber/transcripts"
    
    # Create backup before renaming
    backup_path = Path(base_dir).parent / "transcripts_backup_before_rename"
    print(f"Creating backup at: {backup_path}")
    
    # Use robocopy for Windows or cp for Unix
    if os.name == 'nt':
        os.system(f'robocopy "{base_dir}" "{backup_path}" /E')
    else:
        os.system(f'cp -r "{base_dir}" "{backup_path}"')
    
    try:
        rename_files(base_dir)
        print("\nFile renaming completed successfully!")
    except Exception as e:
        print(f"\nError during renaming: {str(e)}")
        print("Restoring from backup...")
        # Restore from backup
        if os.name == 'nt':
            os.system(f'robocopy "{backup_path}" "{base_dir}" /E')
        else:
            os.system(f'rm -rf "{base_dir}" && cp -r "{backup_path}" "{base_dir}"')
        print("Backup restored.")

if __name__ == "__main__":
    main()