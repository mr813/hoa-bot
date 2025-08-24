"""
User Management System for HOA Bot
Handles authentication, user registration, property management, and email notifications.
"""

import os
import yaml
import bcrypt
import smtplib
import secrets
import string
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import streamlit as st
import streamlit_authenticator as stauth
from datetime import datetime, timedelta
import json


class UserManager:
    """Manages user authentication, registration, and property management."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # File paths
        self.users_file = self.data_dir / "users.yaml"
        self.properties_file = self.data_dir / "properties.json"
        self.user_properties_file = self.data_dir / "user_properties.json"
        
        # Email configuration
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.smtp_username = os.getenv('SMTP_USERNAME')
        self.smtp_password = os.getenv('SMTP_PASSWORD')
        self.from_email = os.getenv('FROM_EMAIL', self.smtp_username)
        
        # Initialize data structures
        self._load_users()
        self._load_properties()
    
    def _load_users(self):
        """Load users from YAML file."""
        if self.users_file.exists():
            with open(self.users_file, 'r') as file:
                self.users = yaml.load(file, Loader=yaml.SafeLoader)
        else:
            self.users = {
                'usernames': {},
                'names': {},
                'emails': {}
            }
            self._save_users()
    
    def _save_users(self):
        """Save users to YAML file."""
        with open(self.users_file, 'w') as file:
            yaml.dump(self.users, file, default_flow_style=False)
    
    def _load_properties(self):
        """Load properties from JSON file."""
        if self.properties_file.exists():
            with open(self.properties_file, 'r') as file:
                self.properties = json.load(file)
        else:
            self.properties = {}
            self._save_properties()
    
    def _save_properties(self):
        """Save properties to JSON file."""
        with open(self.properties_file, 'w') as file:
            json.dump(self.properties, file, indent=2)
    
    def _load_user_properties(self):
        """Load user-property relationships."""
        if self.user_properties_file.exists():
            with open(self.user_properties_file, 'r') as file:
                return json.load(file)
        return {}
    
    def _save_user_properties(self, user_properties: Dict):
        """Save user-property relationships."""
        with open(self.user_properties_file, 'w') as file:
            json.dump(user_properties, file, indent=2)
    
    def _generate_password(self, length: int = 12) -> str:
        """Generate a secure random password."""
        characters = string.ascii_letters + string.digits + "!@#$%^&*"
        return ''.join(secrets.choice(characters) for _ in range(length))
    
    def _send_email(self, to_email: str, subject: str, body: str) -> bool:
        """Send email using configured SMTP settings."""
        if not all([self.smtp_username, self.smtp_password]):
            st.error("Email configuration not set. Please configure SMTP settings.")
            return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = to_email
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.smtp_username, self.smtp_password)
            
            text = msg.as_string()
            server.sendmail(self.from_email, to_email, text)
            server.quit()
            
            return True
        except Exception as e:
            st.error(f"Failed to send email: {str(e)}")
            return False
    
    def register_user(self, name: str, email: str) -> Tuple[bool, str, Optional[str]]:
        """Register a new user and send credentials via email."""
        # Check if email already exists
        if email in self.users['emails']:
            return False, "Email already registered", None
        
        # Generate username from email
        username = email.split('@')[0]
        base_username = username
        counter = 1
        
        while username in self.users['usernames']:
            username = f"{base_username}{counter}"
            counter += 1
        
        # Generate password
        password = self._generate_password()
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        # Add user to data structure
        self.users['usernames'][username] = {
            'name': name,
            'email': email,
            'password': hashed_password.decode('utf-8')
        }
        self.users['names'][name] = username
        self.users['emails'][email] = username
        
        # Save to file
        self._save_users()
        
        # Send email with credentials
        subject = "Welcome to HOA Bot - Your Account Credentials"
        body = f"""
Dear {name},

Welcome to HOA Bot! Your account has been successfully created.

Your login credentials:
Username: {username}
Password: {password}

Please keep these credentials safe. You can change your password after logging in.

Best regards,
The HOA Bot Team
        """
        
        email_sent = self._send_email(email, subject, body)
        
        if email_sent:
            return True, f"Account created successfully! Credentials sent to {email}", username
        else:
            return True, f"Account created but email delivery failed. Username: {username}, Password: {password}", username
    
    def authenticate_user(self, username: str, password: str) -> Tuple[bool, str]:
        """Authenticate a user with username and password."""
        if username not in self.users['usernames']:
            return False, "Invalid username or password"
        
        user_data = self.users['usernames'][username]
        stored_password = user_data['password']
        
        if bcrypt.checkpw(password.encode('utf-8'), stored_password.encode('utf-8')):
            return True, "Authentication successful"
        else:
            return False, "Invalid username or password"
    
    def get_user_info(self, username: str) -> Optional[Dict]:
        """Get user information by username."""
        if username in self.users['usernames']:
            user_data = self.users['usernames'][username].copy()
            user_data['username'] = username
            user_data.pop('password', None)  # Don't return password
            return user_data
        return None
    
    def add_property(self, username: str, address: str, nickname: str, property_type: str) -> Tuple[bool, str]:
        """Add a property to a user's profile."""
        if property_type not in ['condo', 'house']:
            return False, "Property type must be 'condo' or 'house'"
        
        # Generate unique property ID
        property_id = f"{username}_{len(self.get_user_properties(username)) + 1}"
        
        # Create property object
        property_data = {
            'id': property_id,
            'address': address,
            'nickname': nickname,
            'property_type': property_type,
            'created_at': datetime.now().isoformat(),
            'owner': username
        }
        
        # Add to properties
        self.properties[property_id] = property_data
        
        # Add to user properties
        user_properties = self._load_user_properties()
        if username not in user_properties:
            user_properties[username] = []
        user_properties[username].append(property_id)
        
        # Save data
        self._save_properties()
        self._save_user_properties(user_properties)
        
        return True, f"Property '{nickname}' added successfully"
    
    def get_user_properties(self, username: str) -> List[Dict]:
        """Get all properties for a user."""
        user_properties = self._load_user_properties()
        if username not in user_properties:
            return []
        
        properties = []
        for property_id in user_properties[username]:
            if property_id in self.properties:
                properties.append(self.properties[property_id])
        
        return properties
    
    def get_property(self, property_id: str) -> Optional[Dict]:
        """Get property by ID."""
        return self.properties.get(property_id)
    
    def delete_property(self, username: str, property_id: str) -> Tuple[bool, str]:
        """Delete a property from user's profile."""
        # Check if property belongs to user
        if property_id not in self.properties:
            return False, "Property not found"
        
        if self.properties[property_id]['owner'] != username:
            return False, "You don't have permission to delete this property"
        
        # Remove from user properties
        user_properties = self._load_user_properties()
        if username in user_properties and property_id in user_properties[username]:
            user_properties[username].remove(property_id)
        
        # Remove from properties
        del self.properties[property_id]
        
        # Save data
        self._save_properties()
        self._save_user_properties(user_properties)
        
        return True, "Property deleted successfully"
    
    def change_password(self, username: str, current_password: str, new_password: str) -> Tuple[bool, str]:
        """Change user password."""
        # Verify current password
        if username not in self.users['usernames']:
            return False, "User not found"
        
        user_data = self.users['usernames'][username]
        stored_password = user_data['password']
        
        if not bcrypt.checkpw(current_password.encode('utf-8'), stored_password.encode('utf-8')):
            return False, "Current password is incorrect"
        
        # Hash new password
        hashed_new_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
        
        # Update password
        self.users['usernames'][username]['password'] = hashed_new_password.decode('utf-8')
        self._save_users()
        
        return True, "Password changed successfully"


def create_user_manager() -> UserManager:
    """Create and return a UserManager instance."""
    return UserManager()


def get_property_storage_path(property_id: str) -> Path:
    """Get the storage path for a specific property's documents."""
    return Path("data") / "properties" / property_id


def ensure_property_storage(property_id: str) -> Path:
    """Ensure storage directory exists for a property."""
    storage_path = get_property_storage_path(property_id)
    storage_path.mkdir(parents=True, exist_ok=True)
    return storage_path
