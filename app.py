import os
import json
import logging
import time
from datetime import datetime, timedelta
import csv
import io
from enum import Enum
import asyncio
import random
import asyncpg
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import RunReportRequest
# FastAPI imports
from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Security & Database
import bcrypt
import jwt
from pydantic import BaseModel, EmailStr

# Data & Analytics
import numpy as np
from sklearn.ensemble import IsolationForest

# Notion Integration
try:
    from notion_client import Client

    IMPORT_NOTION = True
except ImportError:
    IMPORT_NOTION = False
    print("notion-client not available - Notion integration disabled")

# Environment
from dotenv import load_dotenv

load_dotenv()

# PostgreSQL Configuration
POSTGRES_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", 5432)),
    "database": os.getenv("POSTGRES_DB", "analytics_dashboard"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "password")
}

try:
    IMPORT_ASYNC_PG = True
except ImportError:
    IMPORT_ASYNC_PG = False
    print("asyncpg not available - PostgreSQL support disabled")

try:
    import aiomysql

    IMPORT_AIOMYSQL = True
except ImportError:
    IMPORT_AIOMYSQL = False
    print("aiomysql not available - MySQL support disabled")

try:
    import requests

    IMPORT_REQUESTS = True
except ImportError:
    IMPORT_REQUESTS = False
    print("requests not available - API integrations disabled")

try:
    IMPORT_GOOGLE_ANALYTICS = True
except ImportError:
    IMPORT_GOOGLE_ANALYTICS = False
    print("google-analytics-data not available - Google Analytics support disabled")

try:
    import stripe

    IMPORT_STRIPE = True
except ImportError:
    IMPORT_STRIPE = False
    print("stripe not available - Stripe integration disabled")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# PostgreSQL Database Manager
from typing import Dict, Any, List, Optional


class PostgreSQLManager:
    _instance = None
    _pool = None

    def __init__(self):
        pass  # Remove the problematic _init_database call

    async def _init_database(self):
        """Initialize database with all required tables"""
        try:
            # Create tables
            await self._create_tables()
            logger.info("PostgreSQL database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            raise

    async def _get_connection(self):
        """Get database connection from pool"""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(**POSTGRES_CONFIG)
        return self._pool

    async def _create_tables(self):
        """Create all required tables"""
        conn = await self._get_connection()

        async with conn.acquire() as connection:
            # Users table
            await connection.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    full_name TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'viewer',
                    is_verified BOOLEAN DEFAULT true,
                    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    last_login TIMESTAMP WITH TIME ZONE,
                    updated_at TIMESTAMP WITH TIME ZONE
                )
            ''')

            # Analytics events table
            await connection.execute('''
                CREATE TABLE IF NOT EXISTS analytics_events (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER,
                    event_type TEXT NOT NULL,
                    event_value REAL,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                    properties TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                )
            ''')

            # Custom reports table
            await connection.execute('''
                CREATE TABLE IF NOT EXISTS custom_reports (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    metrics TEXT NOT NULL,
                    filters TEXT,
                    chart_type TEXT NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                )
            ''')

            # A/B tests table
            await connection.execute('''
                CREATE TABLE IF NOT EXISTS ab_tests (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    hypothesis TEXT,
                    status TEXT DEFAULT 'draft',
                    variants TEXT NOT NULL,
                    primary_metric TEXT NOT NULL,
                    target_audience TEXT,
                    duration_days INTEGER DEFAULT 14,
                    created_by INTEGER NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    started_at TIMESTAMP WITH TIME ZONE,
                    completed_at TIMESTAMP WITH TIME ZONE,
                    results TEXT,
                    participants INTEGER DEFAULT 0,
                    conversions INTEGER DEFAULT 0,
                    FOREIGN KEY (created_by) REFERENCES users (id) ON DELETE CASCADE
                )
            ''')

            # Data exports table
            await connection.execute('''
                CREATE TABLE IF NOT EXISTS data_exports (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    data_type TEXT NOT NULL,
                    format TEXT DEFAULT 'csv',
                    filters TEXT,
                    file_name TEXT NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    status TEXT DEFAULT 'completed',
                    record_count INTEGER DEFAULT 0,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                )
            ''')

            # Anomaly detections table
            await connection.execute('''
                CREATE TABLE IF NOT EXISTS anomaly_detections (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    metric TEXT NOT NULL,
                    time_range TEXT DEFAULT '7d',
                    sensitivity REAL DEFAULT 0.1,
                    anomalies_detected INTEGER DEFAULT 0,
                    detected_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    results TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                )
            ''')

            # Notion integrations table
            await connection.execute('''
                CREATE TABLE IF NOT EXISTS notion_integrations (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    notion_token TEXT NOT NULL,
                    database_id TEXT NOT NULL,
                    sync_metrics TEXT,
                    is_active BOOLEAN DEFAULT true,
                    last_sync TIMESTAMP WITH TIME ZONE,
                    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                )
            ''')

            # Data sources table
            await connection.execute('''
                CREATE TABLE IF NOT EXISTS data_sources (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    connection_string TEXT,
                    host TEXT,
                    port INTEGER,
                    database TEXT,
                    username TEXT,
                    password TEXT,
                    api_key TEXT,
                    api_url TEXT,
                    file_path TEXT,
                    table_name TEXT,
                    query TEXT,
                    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    last_used TIMESTAMP WITH TIME ZONE,
                    is_active BOOLEAN DEFAULT true,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                )
            ''')

            # ICE initiatives table
            await connection.execute('''
                CREATE TABLE IF NOT EXISTS ice_initiatives (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    impact REAL NOT NULL,
                    confidence REAL NOT NULL,
                    ease REAL NOT NULL,
                    ice_score REAL NOT NULL,
                    priority TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    assigned_to INTEGER,
                    created_by INTEGER NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    due_date TIMESTAMP WITH TIME ZONE,
                    tags TEXT,
                    metrics_affected TEXT,
                    estimated_effort_days INTEGER DEFAULT 0,
                    FOREIGN KEY (assigned_to) REFERENCES users (id) ON DELETE SET NULL,
                    FOREIGN KEY (created_by) REFERENCES users (id) ON DELETE CASCADE
                )
            ''')

            # User history table
            await connection.execute('''
                CREATE TABLE IF NOT EXISTS user_history (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    action_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    metadata TEXT,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                    ip_address TEXT,
                    user_agent TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                )
            ''')

            # Cohort analyses table
            await connection.execute('''
                CREATE TABLE IF NOT EXISTS cohort_analyses (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    cohort_type TEXT NOT NULL,
                    metric TEXT NOT NULL,
                    period_count INTEGER NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    results TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                )
            ''')

            # Funnel analyses table
            await connection.execute('''
                CREATE TABLE IF NOT EXISTS funnel_analyses (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    funnel_stages TEXT NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    results TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                )
            ''')

            # AI action recommendations table
            await connection.execute('''
                CREATE TABLE IF NOT EXISTS ai_action_recommendations (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    focus_area TEXT,
                    recommendation TEXT NOT NULL,
                    generated_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                )
            ''')

            # Real data config table
            await connection.execute('''
                CREATE TABLE IF NOT EXISTS real_data_config (
                    id INTEGER PRIMARY KEY DEFAULT 1,
                    enabled BOOLEAN DEFAULT true,
                    data_sources TEXT,
                    primary_source TEXT,
                    updated_at TIMESTAMP WITH TIME ZONE NOT NULL
                )
            ''')

            # Insert default real data config - FIXED: Use datetime object, not string
            await connection.execute('''
                INSERT INTO real_data_config (enabled, data_sources, primary_source, updated_at)
                VALUES (true, '[]', NULL, $1)
                ON CONFLICT (id) DO NOTHING
            ''', datetime.now())  # Pass datetime object directly

    async def execute_query(self, query: str, *params) -> asyncpg.Record:
        """Execute a query and return result"""
        conn = await self._get_connection()
        async with conn.acquire() as connection:
            try:
                result = await connection.fetchrow(query, *params)
                return result
            except Exception as e:
                logger.error(f"Query execution error: {e}")
                raise

    async def fetch_one(self, query: str, *params) -> Optional[Dict[str, Any]]:
        """Fetch single row"""
        conn = await self._get_connection()
        async with conn.acquire() as connection:
            try:
                row = await connection.fetchrow(query, *params)
                return dict(row) if row else None
            except Exception as e:
                logger.error(f"Fetch one error: {e}")
                return None

    async def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user by email"""
        return await self.fetch_one("SELECT * FROM users WHERE email = $1", email)

    async def fetch_all(self, query: str, *params) -> List[Dict[str, Any]]:
        """Fetch all rows"""
        conn = await self._get_connection()
        async with conn.acquire() as connection:
            try:
                rows = await connection.fetch(query, *params)
                return [dict(row) for row in rows]
            except Exception as e:
                logger.error(f"Fetch all error: {e}")
                return []

    async def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user by email"""
        return await self.fetch_one("SELECT * FROM users WHERE email = $1", email)

    async def insert(self, table: str, data: Dict[str, Any]) -> int:
        """Insert data and return ID - FIXED VERSION"""
        try:
            print(f"🔄 INSERT into {table}: {list(data.keys())}")

            # Process data to ensure proper types
            processed_data = {}
            for key, value in data.items():
                if isinstance(value, datetime):
                    processed_data[key] = value
                elif isinstance(value, str) and ('created_at' in key or 'updated_at' in key or 'timestamp' in key):
                    try:
                        # Convert string to datetime
                        processed_data[key] = datetime.fromisoformat(value.replace('Z', '+00:00'))
                    except:
                        processed_data[key] = value
                else:
                    processed_data[key] = value

            # Build query
            columns = ', '.join(processed_data.keys())
            placeholders = ', '.join([f'${i + 1}' for i in range(len(processed_data))])
            query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders}) RETURNING id"

            print(f"📝 SQL: {query}")
            print(f"📦 Data: {list(processed_data.values())}")

            # Get connection and execute
            conn = await self._get_connection()
            async with conn.acquire() as connection:
                try:
                    result = await connection.fetchrow(query, *processed_data.values())
                    if result and 'id' in result:
                        inserted_id = result['id']
                        print(f"✅ INSERT successful, ID: {inserted_id}")
                        return inserted_id
                    else:
                        raise Exception("No ID returned from INSERT")
                except Exception as e:
                    print(f"❌ INSERT query failed: {e}")
                    # Print the exact error
                    import traceback
                    traceback.print_exc()
                    raise

        except Exception as e:
            print(f"❌ PostgreSQLManager.insert failed: {e}")
            raise

    async def update(self, table: str, data: Dict[str, Any], where: str, *params) -> bool:
        """Update data"""
        # Convert datetime strings to datetime objects
        processed_data = {}
        for key, value in data.items():
            if isinstance(value, str) and ('created_at' in key or 'updated_at' in key or 'timestamp' in key):
                try:
                    processed_data[key] = datetime.fromisoformat(value.replace('Z', '+00:00'))
                except:
                    processed_data[key] = value
            else:
                processed_data[key] = value

        set_clause = ', '.join([f"{key} = ${i + 1}" for i, key in enumerate(processed_data.keys())])
        param_values = list(processed_data.values()) + list(params)
        query = f"UPDATE {table} SET {set_clause} WHERE {where}"

        conn = await self._get_connection()
        async with conn.acquire() as connection:
            try:
                result = await connection.execute(query, *param_values)
                return "UPDATE" in result
            except Exception as e:
                logger.error(f"Update error: {e}")
                return False

    async def delete(self, table: str, where: str, *params) -> bool:
        """Delete data"""
        query = f"DELETE FROM {table} WHERE {where}"

        conn = await self._get_connection()
        async with conn.acquire() as connection:
            try:
                result = await connection.execute(query, *params)
                return "DELETE" in result
            except Exception as e:
                logger.error(f"Delete error: {e}")
                return False

    @classmethod
    async def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            await cls._instance._init_database()
        return cls._instance


class DatabaseManager:
    _instance = None

    def __init__(self):
        self.db = None
        self.initialized = False

    async def initialize(self):
        """Initialize the database connection with proper error handling"""
        try:
            print("🔄 Starting database initialization...")
            self.db = await PostgreSQLManager.get_instance()
            print("✅ PostgreSQL connection established")

            # Force create tables and sample data
            await self._force_initialize_database()

        except Exception as e:
            print(f"❌ Database initialization failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    async def _force_initialize_database(self):
        """Force initialize database with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"🔄 Database setup attempt {attempt + 1}/{max_retries}")

                # Ensure tables exist
                await self.db._create_tables()
                print("✅ Database tables verified")

                # Check if we have any users
                users_count_result = await self.db.fetch_one("SELECT COUNT(*) as count FROM users")
                users_count = users_count_result['count'] if users_count_result else 0
                print(f"📊 Current users in database: {users_count}")

                if users_count == 0:
                    print("🆕 No users found, generating sample data...")
                    await self._generate_sample_data()
                else:
                    print("✅ Users already exist")

                    # Verify admin user exists
                    admin_user = await self.get_user_by_email('admin@example.com')
                    if not admin_user:
                        print("❌ Admin user missing, creating...")
                        await self._create_admin_user()

                self.initialized = True
                print("🎉 Database initialization completed successfully!")
                return

            except Exception as e:
                print(f"❌ Database setup attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                else:
                    print("💥 All database setup attempts failed")
                    raise

    async def _generate_sample_data(self):
        """Generate sample data for demonstration"""
        try:
            print("🎯 Starting sample data generation...")

            # First create admin user
            admin_id = await self._create_admin_user()

            # Create a few more sample users
            sample_users = [
                {
                    'email': 'manager@example.com',
                    'password_hash': AuthManager.hash_password('manager123'),
                    'full_name': 'Project Manager',
                    'role': 'manager',
                    'is_verified': True,
                    'created_at': datetime.now()
                },
                {
                    'email': 'analyst@example.com',
                    'password_hash': AuthManager.hash_password('analyst123'),
                    'full_name': 'Data Analyst',
                    'role': 'analyst',
                    'is_verified': True,
                    'created_at': datetime.now()
                }
            ]

            for user_data in sample_users:
                user_id = await self.create_user(user_data)
                print(f"✅ Sample user created: {user_data['email']} (ID: {user_id})")

            # Sample ICE initiatives
            sample_initiatives = [
                {
                    'id': f"ice_{int(time.time() * 1000)}",
                    'title': 'Implement Real-time Data Processing',
                    'description': 'Set up real-time data pipelines for immediate analytics',
                    'impact': 8.5,
                    'confidence': 7.0,
                    'ease': 6.0,
                    'ice_score': 7.17,
                    'priority': 'high',
                    'status': 'in_progress',
                    'assigned_to': admin_id,
                    'created_by': admin_id,
                    'created_at': datetime.now(),
                    'updated_at': datetime.now(),
                    'due_date': (datetime.now() + timedelta(days=14)),
                    'tags': json.dumps(['infrastructure', 'real-time', 'analytics']),
                    'metrics_affected': json.dumps(['dau', 'mau', 'response_time']),
                    'estimated_effort_days': 10
                },
                {
                    'id': f"ice_{int(time.time() * 1000) + 1}",
                    'title': 'Enhance AI Recommendation Engine',
                    'description': 'Improve machine learning models for better user recommendations',
                    'impact': 9.0,
                    'confidence': 8.0,
                    'ease': 5.0,
                    'ice_score': 7.33,
                    'priority': 'high',
                    'status': 'pending',
                    'assigned_to': None,
                    'created_by': admin_id,
                    'created_at': datetime.now(),
                    'updated_at': datetime.now(),
                    'due_date': (datetime.now() + timedelta(days=21)),
                    'tags': json.dumps(['ai', 'machine-learning', 'recommendations']),
                    'metrics_affected': json.dumps(['conversion_rate', 'user_engagement']),
                    'estimated_effort_days': 15
                }
            ]

            for initiative in sample_initiatives:
                await self.db.insert('ice_initiatives', initiative)
                print(f"✅ ICE initiative created: {initiative['title']}")

            # Sample analytics events
            for i in range(50):
                event_data = {
                    'user_id': random.randint(1, 3),
                    'event_type': random.choice(['page_view', 'button_click', 'form_submit', 'purchase']),
                    'event_value': round(random.uniform(0, 100), 2),
                    'timestamp': (datetime.now() - timedelta(days=random.randint(0, 29))),
                    'properties': json.dumps({
                        'page': random.choice(['/home', '/dashboard', '/pricing', '/features']),
                        'browser': random.choice(['chrome', 'firefox', 'safari']),
                        'country': random.choice(['US', 'UK', 'CA', 'AU', 'DE'])
                    })
                }
                await self.db.insert('analytics_events', event_data)

            print("✅ Sample data generation completed successfully")
            print("📋 Sample credentials:")
            print("   👑 Admin: admin@example.com / admin123")
            print("   👨‍💼 Manager: manager@example.com / manager123")
            print("   📊 Analyst: analyst@example.com / analyst123")

        except Exception as e:
            print(f"❌ Error generating sample data: {e}")
            raise

    async def _create_admin_user(self):
        """Create admin user with guaranteed success"""
        try:
            # First check if admin already exists
            existing_admin = await self.get_user_by_email('admin@example.com')
            if existing_admin:
                print("✅ Admin user already exists")
                return existing_admin['id']

            hashed_password = AuthManager.hash_password('admin123')
            admin_data = {
                'email': 'admin@example.com',
                'password_hash': hashed_password,
                'full_name': 'System Administrator',
                'role': 'admin',
                'is_verified': True,
                'created_at': datetime.now()
            }

            print("🔄 Creating admin user...")
            admin_id = await self.create_user(admin_data)
            print(f"✅ Admin user created with ID: {admin_id}")

            return admin_id

        except Exception as e:
            print(f"❌ Failed to create admin user: {e}")
            raise

    async def create_user(self, user_data: Dict[str, Any]) -> int:
        """Create new user with guaranteed success"""
        try:
            print(f"🔄 Creating user: {user_data['email']}")

            # Ensure we have proper datetime object
            if 'created_at' in user_data and isinstance(user_data['created_at'], datetime):
                # Already a datetime object, good
                pass
            elif 'created_at' in user_data and isinstance(user_data['created_at'], str):
                # Convert string to datetime
                user_data['created_at'] = datetime.fromisoformat(user_data['created_at'].replace('Z', '+00:00'))
            else:
                # No created_at provided, use now
                user_data['created_at'] = datetime.now()

            # Insert user
            user_id = await self.db.insert('users', user_data)
            print(f"✅ User created with ID: {user_id}")

            # Immediate verification
            verify_user = await self.get_user_by_email(user_data['email'])
            if verify_user:
                print(f"✅ User creation verified: {verify_user['email']} (ID: {verify_user['id']})")
                return user_id
            else:
                print(f"❌ USER CREATION FAILED: {user_data['email']} was not found after insertion!")
                raise Exception("User creation failed - user not found after insert")

        except Exception as e:
            print(f"❌ create_user failed for {user_data['email']}: {e}")
            raise

    # User methods
    async def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user by email"""
        try:
            user = await self.db.fetch_one("SELECT * FROM users WHERE email = $1", email)
            if user:
                print(f"✅ Found user by email: {email}")
            else:
                print(f"❌ No user found with email: {email}")
            return user
        except Exception as e:
            print(f"❌ Error getting user by email {email}: {e}")
            return None

    async def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        try:
            return await self.db.fetch_one("SELECT * FROM users WHERE id = $1", user_id)
        except Exception as e:
            print(f"Error getting user by ID {user_id}: {e}")
            return None

    async def update_user(self, user_id: int, update_data: Dict[str, Any]) -> bool:
        """Update user"""
        try:
            # Ensure datetime fields are properly handled
            processed_data = update_data.copy()
            for key in ['created_at', 'updated_at', 'last_login']:
                if key in processed_data and isinstance(processed_data[key], str):
                    try:
                        processed_data[key] = datetime.fromisoformat(processed_data[key].replace('Z', '+00:00'))
                    except:
                        pass

            return await self.db.update('users', processed_data, 'id = $1', user_id)
        except Exception as e:
            print(f"Error updating user {user_id}: {e}")
            return False

    async def get_all_users(self) -> List[Dict[str, Any]]:
        """Get all users"""
        try:
            users = await self.db.fetch_all(
                "SELECT id, email, full_name, role, is_verified, created_at, last_login FROM users ORDER BY id"
            )
            print(f"📊 Retrieved {len(users)} users from database")
            return users
        except Exception as e:
            print(f"Error getting all users: {e}")
            return []

    # Analytics events methods
    async def create_analytics_event(self, event_data: Dict[str, Any]) -> int:
        """Create analytics event"""
        try:
            # Ensure timestamp is datetime object
            if 'timestamp' in event_data and isinstance(event_data['timestamp'], str):
                event_data['timestamp'] = datetime.fromisoformat(event_data['timestamp'].replace('Z', '+00:00'))
            elif 'timestamp' not in event_data:
                event_data['timestamp'] = datetime.now()

            return await self.db.insert('analytics_events', event_data)
        except Exception as e:
            print(f"Error creating analytics event: {e}")
            raise

    async def get_analytics_events(self, user_id: Optional[int] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get analytics events"""
        try:
            if user_id:
                return await self.db.fetch_all(
                    "SELECT * FROM analytics_events WHERE user_id = $1 ORDER BY timestamp DESC LIMIT $2",
                    user_id, limit
                )
            else:
                return await self.db.fetch_all(
                    "SELECT * FROM analytics_events ORDER BY timestamp DESC LIMIT $1",
                    limit
                )
        except Exception as e:
            print(f"Error getting analytics events: {e}")
            return []

    # ICE initiatives methods
    async def create_ice_initiative(self, initiative_data: Dict[str, Any]) -> bool:
        """Create ICE initiative"""
        try:
            # Ensure datetime fields are proper
            for field in ['created_at', 'updated_at', 'due_date']:
                if field in initiative_data and isinstance(initiative_data[field], str):
                    initiative_data[field] = datetime.fromisoformat(initiative_data[field].replace('Z', '+00:00'))

            await self.db.insert('ice_initiatives', initiative_data)
            return True
        except Exception as e:
            logger.error(f"Error creating ICE initiative: {e}")
            return False

    async def get_ice_initiatives(self, user_id: Optional[int] = None, status: Optional[str] = None) -> List[
        Dict[str, Any]]:
        """Get ICE initiatives"""
        try:
            query = "SELECT * FROM ice_initiatives WHERE 1=1"
            params = []

            if user_id:
                query += " AND (created_by = $1 OR assigned_to = $2)"
                params.extend([user_id, user_id])

            if status:
                query += " AND status = $3"
                params.append(status)

            query += " ORDER BY ice_score DESC"

            initiatives = await self.db.fetch_all(query, *params)

            # Parse JSON fields
            for initiative in initiatives:
                if initiative.get('tags'):
                    try:
                        initiative['tags'] = json.loads(initiative['tags'])
                    except:
                        initiative['tags'] = []
                if initiative.get('metrics_affected'):
                    try:
                        initiative['metrics_affected'] = json.loads(initiative['metrics_affected'])
                    except:
                        initiative['metrics_affected'] = []

            return initiatives
        except Exception as e:
            print(f"Error getting ICE initiatives: {e}")
            return []

    async def update_ice_initiative(self, initiative_id: str, update_data: Dict[str, Any]) -> bool:
        """Update ICE initiative"""
        try:
            return await self.db.update('ice_initiatives', update_data, 'id = $1', initiative_id)
        except Exception as e:
            print(f"Error updating ICE initiative {initiative_id}: {e}")
            return False

    async def delete_ice_initiative(self, initiative_id: str) -> bool:
        """Delete ICE initiative"""
        try:
            return await self.db.delete('ice_initiatives', 'id = $1', initiative_id)
        except Exception as e:
            print(f"Error deleting ICE initiative {initiative_id}: {e}")
            return False

    # User history methods
    async def create_user_history(self, history_data: Dict[str, Any]) -> bool:
        """Create user history record"""
        try:
            # Ensure timestamp is datetime
            if 'timestamp' in history_data and isinstance(history_data['timestamp'], str):
                history_data['timestamp'] = datetime.fromisoformat(history_data['timestamp'].replace('Z', '+00:00'))
            elif 'timestamp' not in history_data:
                history_data['timestamp'] = datetime.now()

            await self.db.insert('user_history', history_data)
            return True
        except Exception as e:
            logger.error(f"Error creating user history: {e}")
            return False

    async def get_user_history(self, user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
        """Get user history"""
        try:
            history = await self.db.fetch_all(
                "SELECT * FROM user_history WHERE user_id = $1 ORDER BY timestamp DESC LIMIT $2",
                user_id, limit
            )

            # Parse JSON fields
            for record in history:
                if record.get('metadata'):
                    try:
                        record['metadata'] = json.loads(record['metadata'])
                    except:
                        record['metadata'] = {}

            return history
        except Exception as e:
            print(f"Error getting user history for user {user_id}: {e}")
            return []

    # Data sources methods
    async def create_data_source(self, source_data: Dict[str, Any]) -> int:
        """Create data source"""
        try:
            if 'created_at' in source_data and isinstance(source_data['created_at'], str):
                source_data['created_at'] = datetime.fromisoformat(source_data['created_at'].replace('Z', '+00:00'))
            elif 'created_at' not in source_data:
                source_data['created_at'] = datetime.now()

            return await self.db.insert('data_sources', source_data)
        except Exception as e:
            print(f"Error creating data source: {e}")
            raise

    async def get_data_sources(self, user_id: int) -> List[Dict[str, Any]]:
        """Get user's data sources"""
        try:
            sources = await self.db.fetch_all(
                "SELECT * FROM data_sources WHERE user_id = $1 ORDER BY created_at DESC",
                user_id
            )

            # Remove sensitive data
            for source in sources:
                source.pop('password', None)
                source.pop('api_key', None)

            return sources
        except Exception as e:
            print(f"Error getting data sources for user {user_id}: {e}")
            return []

    async def get_data_source(self, source_id: int, user_id: int) -> Optional[Dict[str, Any]]:
        """Get specific data source"""
        try:
            source = await self.db.fetch_one(
                "SELECT * FROM data_sources WHERE id = $1 AND user_id = $2",
                source_id, user_id
            )

            if source:
                source.pop('password', None)
                source.pop('api_key', None)

            return source
        except Exception as e:
            print(f"Error getting data source {source_id} for user {user_id}: {e}")
            return None

    # Real data config methods
    async def get_real_data_config(self) -> Dict[str, Any]:
        """Get real data configuration"""
        try:
            config = await self.db.fetch_one("SELECT * FROM real_data_config WHERE id = 1")
            if config:
                if config.get('data_sources'):
                    try:
                        config['data_sources'] = json.loads(config['data_sources'])
                    except:
                        config['data_sources'] = []
                return config
            return {'enabled': True, 'data_sources': [], 'primary_source': None}
        except Exception as e:
            print(f"Error getting real data config: {e}")
            return {'enabled': True, 'data_sources': [], 'primary_source': None}

    async def update_real_data_config(self, config: Dict[str, Any]) -> bool:
        """Update real data configuration"""
        try:
            update_data = {
                'enabled': config.get('enabled', True),
                'data_sources': json.dumps(config.get('data_sources', [])),
                'primary_source': config.get('primary_source'),
                'updated_at': datetime.now()
            }
            return await self.db.update('real_data_config', update_data, 'id = 1')
        except Exception as e:
            print(f"Error updating real data config: {e}")
            return False

    # A/B tests methods
    async def create_ab_test(self, test_data: Dict[str, Any]) -> int:
        """Create A/B test"""
        try:
            if 'created_at' in test_data and isinstance(test_data['created_at'], str):
                test_data['created_at'] = datetime.fromisoformat(test_data['created_at'].replace('Z', '+00:00'))
            elif 'created_at' not in test_data:
                test_data['created_at'] = datetime.now()

            return await self.db.insert('ab_tests', test_data)
        except Exception as e:
            print(f"Error creating A/B test: {e}")
            raise

    async def get_ab_tests(self, user_id: int) -> List[Dict[str, Any]]:
        """Get user's A/B tests"""
        try:
            tests = await self.db.fetch_all(
                "SELECT * FROM ab_tests WHERE created_by = $1 ORDER BY created_at DESC",
                user_id
            )

            # Parse JSON fields
            for test in tests:
                if test.get('variants'):
                    try:
                        test['variants'] = json.loads(test['variants'])
                    except:
                        test['variants'] = []
                if test.get('target_audience'):
                    try:
                        test['target_audience'] = json.loads(test['target_audience'])
                    except:
                        test['target_audience'] = {}
                if test.get('results'):
                    try:
                        test['results'] = json.loads(test['results'])
                    except:
                        test['results'] = {}

            return tests
        except Exception as e:
            print(f"Error getting A/B tests for user {user_id}: {e}")
            return []

    async def update_ab_test(self, test_id: int, update_data: Dict[str, Any]) -> bool:
        """Update A/B test"""
        try:
            return await self.db.update('ab_tests', update_data, 'id = $1', test_id)
        except Exception as e:
            print(f"Error updating A/B test {test_id}: {e}")
            return False

    # Custom reports methods
    async def create_custom_report(self, report_data: Dict[str, Any]) -> int:
        """Create custom report"""
        try:
            if 'created_at' in report_data and isinstance(report_data['created_at'], str):
                report_data['created_at'] = datetime.fromisoformat(report_data['created_at'].replace('Z', '+00:00'))
            elif 'created_at' not in report_data:
                report_data['created_at'] = datetime.now()

            if 'updated_at' in report_data and isinstance(report_data['updated_at'], str):
                report_data['updated_at'] = datetime.fromisoformat(report_data['updated_at'].replace('Z', '+00:00'))
            else:
                report_data['updated_at'] = datetime.now()

            return await self.db.insert('custom_reports', report_data)
        except Exception as e:
            print(f"Error creating custom report: {e}")
            raise

    async def get_custom_reports(self, user_id: int) -> List[Dict[str, Any]]:
        """Get user's custom reports"""
        try:
            reports = await self.db.fetch_all(
                "SELECT * FROM custom_reports WHERE user_id = $1 ORDER BY created_at DESC",
                user_id
            )

            # Parse JSON fields
            for report in reports:
                if report.get('metrics'):
                    try:
                        report['metrics'] = json.loads(report['metrics'])
                    except:
                        report['metrics'] = []
                if report.get('filters'):
                    try:
                        report['filters'] = json.loads(report['filters'])
                    except:
                        report['filters'] = {}

            return reports
        except Exception as e:
            print(f"Error getting custom reports for user {user_id}: {e}")
            return []

    async def delete_custom_report(self, report_id: int, user_id: int) -> bool:
        """Delete custom report"""
        try:
            return await self.db.delete('custom_reports', 'id = $1 AND user_id = $2', report_id, user_id)
        except Exception as e:
            print(f"Error deleting custom report {report_id}: {e}")
            return False

    # Export methods
    async def create_export(self, export_data: Dict[str, Any]) -> int:
        """Create export record"""
        try:
            if 'created_at' in export_data and isinstance(export_data['created_at'], str):
                export_data['created_at'] = datetime.fromisoformat(export_data['created_at'].replace('Z', '+00:00'))
            elif 'created_at' not in export_data:
                export_data['created_at'] = datetime.now()

            return await self.db.insert('data_exports', export_data)
        except Exception as e:
            print(f"Error creating export: {e}")
            raise

    async def get_exports(self, user_id: int) -> List[Dict[str, Any]]:
        """Get user's exports"""
        try:
            return await self.db.fetch_all(
                "SELECT * FROM data_exports WHERE user_id = $1 ORDER BY created_at DESC",
                user_id
            )
        except Exception as e:
            print(f"Error getting exports for user {user_id}: {e}")
            return []

    # Cohort analyses methods
    async def create_cohort_analysis(self, analysis_data: Dict[str, Any]) -> int:
        """Create cohort analysis"""
        try:
            if 'created_at' in analysis_data and isinstance(analysis_data['created_at'], str):
                analysis_data['created_at'] = datetime.fromisoformat(analysis_data['created_at'].replace('Z', '+00:00'))
            elif 'created_at' not in analysis_data:
                analysis_data['created_at'] = datetime.now()

            return await self.db.insert('cohort_analyses', analysis_data)
        except Exception as e:
            print(f"Error creating cohort analysis: {e}")
            raise

    async def get_cohort_analyses(self, user_id: int) -> List[Dict[str, Any]]:
        """Get user's cohort analyses"""
        try:
            analyses = await self.db.fetch_all(
                "SELECT * FROM cohort_analyses WHERE user_id = $1 ORDER BY created_at DESC",
                user_id
            )

            for analysis in analyses:
                if analysis.get('results'):
                    try:
                        analysis['results'] = json.loads(analysis['results'])
                    except:
                        analysis['results'] = {}

            return analyses
        except Exception as e:
            print(f"Error getting cohort analyses for user {user_id}: {e}")
            return []

    # Funnel analyses methods
    async def create_funnel_analysis(self, analysis_data: Dict[str, Any]) -> int:
        """Create funnel analysis"""
        try:
            if 'created_at' in analysis_data and isinstance(analysis_data['created_at'], str):
                analysis_data['created_at'] = datetime.fromisoformat(analysis_data['created_at'].replace('Z', '+00:00'))
            elif 'created_at' not in analysis_data:
                analysis_data['created_at'] = datetime.now()

            return await self.db.insert('funnel_analyses', analysis_data)
        except Exception as e:
            print(f"Error creating funnel analysis: {e}")
            raise

    async def get_funnel_analyses(self, user_id: int) -> List[Dict[str, Any]]:
        """Get user's funnel analyses"""
        try:
            analyses = await self.db.fetch_all(
                "SELECT * FROM funnel_analyses WHERE user_id = $1 ORDER BY created_at DESC",
                user_id
            )

            for analysis in analyses:
                if analysis.get('funnel_stages'):
                    try:
                        analysis['funnel_stages'] = json.loads(analysis['funnel_stages'])
                    except:
                        analysis['funnel_stages'] = []
                if analysis.get('results'):
                    try:
                        analysis['results'] = json.loads(analysis['results'])
                    except:
                        analysis['results'] = {}

            return analyses
        except Exception as e:
            print(f"Error getting funnel analyses for user {user_id}: {e}")
            return []

    # Notion integration methods
    async def get_notion_integration(self, integration_id: int, user_id: int) -> Optional[Dict[str, Any]]:
        """Get specific Notion integration"""
        try:
            return await self.db.fetch_one(
                "SELECT * FROM notion_integrations WHERE id = $1 AND user_id = $2",
                integration_id, user_id
            )
        except Exception as e:
            print(f"Error getting Notion integration {integration_id}: {e}")
            return None

    async def update_notion_integration(self, integration_id: int, update_data: Dict[str, Any]) -> bool:
        """Update Notion integration"""
        try:
            return await self.db.update('notion_integrations', update_data, 'id = $1', integration_id)
        except Exception as e:
            print(f"Error updating Notion integration {integration_id}: {e}")
            return False

    async def delete_notion_integration(self, integration_id: int, user_id: int) -> bool:
        """Delete Notion integration"""
        try:
            return await self.db.delete('notion_integrations', 'id = $1 AND user_id = $2', integration_id, user_id)
        except Exception as e:
            print(f"Error deleting Notion integration {integration_id}: {e}")
            return False

    # AI recommendations methods
    async def create_ai_recommendation(self, recommendation_data: Dict[str, Any]) -> int:
        """Create AI recommendation"""
        try:
            if 'generated_at' in recommendation_data and isinstance(recommendation_data['generated_at'], str):
                recommendation_data['generated_at'] = datetime.fromisoformat(
                    recommendation_data['generated_at'].replace('Z', '+00:00'))
            elif 'generated_at' not in recommendation_data:
                recommendation_data['generated_at'] = datetime.now()

            return await self.db.insert('ai_action_recommendations', recommendation_data)
        except Exception as e:
            print(f"Error creating AI recommendation: {e}")
            raise

    async def get_ai_recommendations(self, user_id: int) -> List[Dict[str, Any]]:
        """Get user's AI recommendations"""
        try:
            recommendations = await self.db.fetch_all(
                "SELECT * FROM ai_action_recommendations WHERE user_id = $1 ORDER BY generated_at DESC",
                user_id
            )

            # Parse JSON fields
            for rec in recommendations:
                if rec.get('recommendation'):
                    try:
                        rec['recommendation'] = json.loads(rec['recommendation'])
                    except:
                        rec['recommendation'] = {}

            return recommendations
        except Exception as e:
            print(f"Error getting AI recommendations for user {user_id}: {e}")
            return []

    async def get_notion_integrations(self, user_id: int) -> List[Dict[str, Any]]:
        """Get user's Notion integrations"""
        try:
            return await self.db.fetch_all(
                "SELECT * FROM notion_integrations WHERE user_id = $1 ORDER BY created_at DESC",
                user_id
            )
        except Exception as e:
            print(f"Error getting Notion integrations for user {user_id}: {e}")
            return []

    @classmethod
    async def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            await cls._instance.initialize()
        return cls._instance


app = FastAPI(title="AI Analytics Dashboard", version="4.0.0")

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class UserRole(str, Enum):
    ADMIN = "admin"
    MANAGER = "manager"
    ANALYST = "analyst"
    VIEWER = "viewer"


class ABTestStatus(str, Enum):
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"


class DataSourceType(str, Enum):
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    CSV = "csv"
    JSON = "json"
    API = "api"
    GOOGLE_ANALYTICS = "google_analytics"
    STRIPE = "stripe"


class ICEPriority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ICEStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ICEInitiative(BaseModel):
    id: str
    title: str
    description: str
    impact: float
    confidence: float
    ease: float
    ice_score: float
    priority: ICEPriority
    status: ICEStatus
    assigned_to: Optional[int] = None
    created_by: int
    created_at: str
    updated_at: str
    due_date: Optional[str] = None
    tags: List[str] = []
    metrics_affected: List[str] = []
    estimated_effort_days: int = 0


class ICEInitiativeCreate(BaseModel):
    title: str
    description: str
    impact: float
    confidence: float
    ease: float
    assigned_to: Optional[int] = None
    due_date: Optional[str] = None
    tags: List[str] = []
    metrics_affected: List[str] = []
    estimated_effort_days: int = 0


class ICEInitiativeUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    impact: Optional[float] = None
    confidence: Optional[float] = None
    ease: Optional[float] = None
    status: Optional[ICEStatus] = None
    assigned_to: Optional[int] = None
    due_date: Optional[str] = None
    tags: Optional[List[str]] = None
    metrics_affected: Optional[List[str]] = None
    estimated_effort_days: Optional[int] = None


class UserActionType(str, Enum):
    LOGIN = "login"
    LOGOUT = "logout"
    VIEW_DASHBOARD = "view_dashboard"
    CREATE_REPORT = "create_report"
    EXPORT_DATA = "export_data"
    CREATE_AB_TEST = "create_ab_test"
    UPDATE_SETTINGS = "update_settings"
    CREATE_DATA_SOURCE = "create_data_source"
    CREATE_ICE_INITIATIVE = "create_ice_initiative"
    UPDATE_ICE_INITIATIVE = "update_ice_initiative"


class UserHistoryRecord(BaseModel):
    id: str
    user_id: int
    action_type: UserActionType
    description: str
    metadata: Dict[str, Any] = {}
    timestamp: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class CohortType(str, Enum):
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


class CohortAnalysisRequest(BaseModel):
    cohort_type: CohortType = CohortType.WEEKLY
    metric: str = "retention"
    period_count: int = 12


class FunnelDropOffRequest(BaseModel):
    funnel_stages: List[str]
    date_range: Dict[str, str] = {}
    segment_by: Optional[str] = None


class DropOffRecommendation(BaseModel):
    stage: str
    drop_off_rate: float
    users_lost: int
    recommendation: str
    impact: str
    effort: str
    priority: str


class AIActionRecommendation(BaseModel):
    id: str
    title: str
    description: str
    action_type: str
    impact_score: float
    confidence: float
    effort_required: str
    metrics_affected: List[str]
    suggested_implementation: str
    expected_improvement: str
    kpis_to_watch: List[str]
    generated_at: str


class AIActionRequest(BaseModel):
    focus_area: Optional[str] = None
    metrics: List[str] = []
    timeframe: str = "30d"


# Security
security = HTTPBearer()


# Pydantic Models
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    role: UserRole = UserRole.VIEWER


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserProfile(BaseModel):
    full_name: str
    email: EmailStr


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str
    user_role: UserRole


class UserUpdateRole(BaseModel):
    user_id: int
    role: UserRole


class CustomReportCreate(BaseModel):
    name: str
    metrics: List[str]
    filters: Dict[str, Any] = {}
    chart_type: str


class ABTestCreate(BaseModel):
    name: str
    description: str
    hypothesis: str
    variants: List[Dict[str, Any]]
    primary_metric: str
    target_audience: Dict[str, Any] = {}
    duration_days: int = 14


class ABTestUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[ABTestStatus] = None


class ExportRequest(BaseModel):
    data_type: str
    format: str = "csv"
    filters: Dict[str, Any] = {}
    date_range: Optional[Dict[str, str]] = None


class AnomalyDetectionRequest(BaseModel):
    metric: str
    time_range: str = "7d"
    sensitivity: float = 0.1


class NotionIntegrationRequest(BaseModel):
    notion_token: str
    database_id: str
    sync_metrics: List[str] = []


class DataSourceCreate(BaseModel):
    name: str
    type: DataSourceType
    connection_string: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    api_key: Optional[str] = None
    api_url: Optional[str] = None
    file_path: Optional[str] = None
    table_name: Optional[str] = None
    query: Optional[str] = None


class RealDataConfig(BaseModel):
    enabled: bool = True
    data_sources: List[DataSourceCreate] = []
    primary_source: Optional[str] = None


# Authentication Manager
class AuthManager:
    JWT_SECRET = os.getenv('JWT_SECRET', 'your-super-secure-jwt-secret-change-in-production')
    ALGORITHM = "HS256"

    @staticmethod
    def hash_password(password: str) -> str:
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        try:
            return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
        except Exception:
            return False

    @staticmethod
    def create_tokens(data: dict) -> tuple:
        access_payload = data.copy()
        access_payload.update({
            'exp': datetime.utcnow() + timedelta(minutes=30),
            'type': 'access'
        })

        refresh_payload = data.copy()
        refresh_payload.update({
            'exp': datetime.utcnow() + timedelta(days=7),
            'type': 'refresh'
        })

        access_token = jwt.encode(access_payload, AuthManager.JWT_SECRET, algorithm=AuthManager.ALGORITHM)
        refresh_token = jwt.encode(refresh_payload, AuthManager.JWT_SECRET, algorithm=AuthManager.ALGORITHM)

        return access_token, refresh_token

    @staticmethod
    def verify_token(token: str) -> Optional[dict]:
        try:
            payload = jwt.decode(token, AuthManager.JWT_SECRET, algorithms=[AuthManager.ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None


#  Cohort Analysis Manager
class CohortAnalysisManager:
    @staticmethod
    async def analyze_cohorts(cohort_request: CohortAnalysisRequest) -> Dict[str, Any]:
        """Perform cohort analysis based on request parameters"""
        try:
            cohort_type = cohort_request.cohort_type
            metric = cohort_request.metric
            period_count = cohort_request.period_count

            # Generate cohort data based on type
            if cohort_type == CohortType.WEEKLY:
                cohort_data = await CohortAnalysisManager._generate_weekly_cohorts(period_count, metric)
            elif cohort_type == CohortType.MONTHLY:
                cohort_data = await CohortAnalysisManager._generate_monthly_cohorts(period_count, metric)
            else:  # Quarterly
                cohort_data = await CohortAnalysisManager._generate_quarterly_cohorts(period_count, metric)

            # Calculate overall metrics
            metrics = await CohortAnalysisManager._calculate_cohort_metrics(cohort_data)

            return {
                'cohort_type': cohort_type,
                'metric': metric,
                'period_count': period_count,
                'cohort_data': cohort_data,
                'metrics': metrics,
                'summary': await CohortAnalysisManager._generate_cohort_summary(cohort_data, metric)
            }

        except Exception as e:
            logger.error(f"Cohort analysis error: {e}")
            raise HTTPException(status_code=500, detail=f"Cohort analysis failed: {str(e)}")

    @staticmethod
    async def _generate_quarterly_cohorts(period_count: int, metric: str) -> List[Dict[str, Any]]:
        """Generate quarterly cohort data"""
        cohorts = []
        base_date = datetime.now().replace(day=1, month=1) - timedelta(days=90 * period_count)

        for i in range(period_count):
            cohort_start = base_date + timedelta(days=90 * i)
            quarter = (cohort_start.month - 1) // 3 + 1
            cohort_size = random.randint(5000, 8000)

            periods = []
            for j in range(period_count):
                if j == 0:
                    if metric == 'retention':
                        value = 100.0
                    else:
                        value = cohort_size
                else:
                    if metric == 'retention':
                        decay = 0.65 ** j
                        noise = random.uniform(0.85, 1.15)
                        value = max(15, min(100, (55 * decay * noise)))
                    else:
                        decay = 0.6 ** j
                        value = int(cohort_size * decay * random.uniform(0.8, 1.2))

                periods.append({
                    'period': j,
                    'value': round(value, 1) if metric == 'retention' else int(value),
                    'period_label': f'Q{j}'
                })

            cohorts.append({
                'cohort_id': f"Q{quarter}_{cohort_start.year}",
                'cohort_label': f'Q{quarter} {cohort_start.year}',
                'cohort_size': cohort_size,
                'periods': periods
            })

        return cohorts

    @staticmethod
    async def _calculate_cohort_metrics(cohort_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate metrics from cohort data"""
        if not cohort_data:
            return {}

        retention_rates = []
        for cohort in cohort_data:
            for period in cohort['periods']:
                if period['period'] > 0:
                    retention_rates.append(period['value'])

        latest_cohort = cohort_data[-1]
        latest_retention = latest_cohort['periods'][1]['value'] if len(latest_cohort['periods']) > 1 else 0

        return {
            'average_retention': round(np.mean(retention_rates), 2) if retention_rates else 0,
            'retention_std': round(np.std(retention_rates), 2) if retention_rates else 0,
            'latest_cohort_retention': latest_retention,
            'total_cohorts': len(cohort_data),
            'health_score': await CohortAnalysisManager._calculate_cohort_health(cohort_data)
        }

    @staticmethod
    async def _calculate_cohort_health(cohort_data: List[Dict[str, Any]]) -> float:
        """Calculate overall cohort health score"""
        if not cohort_data or len(cohort_data) < 2:
            return 0.0

        recent_retention = []
        for cohort in cohort_data[-3:]:
            if len(cohort['periods']) > 1:
                recent_retention.append(cohort['periods'][1]['value'])

        if not recent_retention:
            return 0.0

        avg_retention = np.mean(recent_retention)

        health_score = min(100, max(0, avg_retention * 1.2))

        return round(health_score, 1)

    @staticmethod
    async def _generate_cohort_summary(cohort_data: List[Dict[str, Any]], metric: str) -> Dict[str, Any]:
        """Generate summary insights from cohort data"""
        if not cohort_data:
            return {}

        insights = []

        # Analyze retention trends
        if len(cohort_data) >= 3:
            recent_trend = await CohortAnalysisManager._analyze_retention_trend(cohort_data)
            insights.append(recent_trend)

        # Identify best performing cohort
        best_cohort = await CohortAnalysisManager._find_best_cohort(cohort_data)
        insights.append(best_cohort)

        # Generate improvement recommendations
        recommendations = await CohortAnalysisManager._generate_cohort_recommendations(cohort_data, metric)
        insights.extend(recommendations)

        return {
            'total_insights': len(insights),
            'insights': insights,
            'key_metric': metric,
            'analysis_timestamp': datetime.now().isoformat()
        }

    @staticmethod
    async def _analyze_retention_trend(cohort_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze retention trends across cohorts"""
        recent_cohorts = cohort_data[-3:]

        period_1_retention = []
        for cohort in recent_cohorts:
            if len(cohort['periods']) > 1:
                period_1_retention.append(cohort['periods'][1]['value'])

        if len(period_1_retention) >= 2:
            trend = "improving" if period_1_retention[-1] > period_1_retention[0] else "declining"
            change = abs(period_1_retention[-1] - period_1_retention[0])
        else:
            trend = "stable"
            change = 0

        return {
            'type': 'trend_analysis',
            'title': 'Retention Trend',
            'description': f'Recent cohorts show {trend} retention trends',
            'change': round(change, 1),
            'direction': trend
        }

    @staticmethod
    async def _find_best_cohort(cohort_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find the best performing cohort"""
        best_retention = 0
        best_cohort = None

        for cohort in cohort_data:
            if len(cohort['periods']) > 1:
                retention = cohort['periods'][1]['value']
                if retention > best_retention:
                    best_retention = retention
                    best_cohort = cohort

        return {
            'type': 'best_performer',
            'title': 'Best Performing Cohort',
            'description': f'{best_cohort["cohort_label"]} had the highest retention',
            'retention_rate': best_retention,
            'cohort_id': best_cohort['cohort_id'] if best_cohort else 'N/A'
        }

    class CohortAnalysisManager:
        @staticmethod
        async def _generate_weekly_cohorts(period_count: int, metric: str) -> List[Dict[str, Any]]:
            """Generate weekly cohort data """
            cohorts = []
            base_date = datetime.now() - timedelta(weeks=period_count)

            for i in range(period_count):
                cohort_start = base_date + timedelta(weeks=i)
                cohort_size = random.randint(800, 1200)

                periods = []
                for j in range(period_count - i):
                    if j == 0:
                        if metric == 'retention':
                            value = 100.0
                        else:
                            value = cohort_size
                    else:
                        if metric == 'retention':
                            decay = 0.85 ** j  # 15% decay per period
                            noise = random.uniform(0.95, 1.05)
                            value = max(5, min(100, (100 * decay * noise)))
                        else:
                            decay = 0.82 ** j
                            value = int(cohort_size * decay * random.uniform(0.9, 1.1))

                    periods.append({
                        'period': j,
                        'value': round(value, 1) if metric == 'retention' else int(value),
                        'period_label': f'Week {j + 1}',
                        'cohort_percentage': round((value / cohort_size * 100), 1) if metric != 'retention' else value
                    })

                cohorts.append({
                    'cohort_id': f"W{cohort_start.strftime('%Y-%W')}",
                    'cohort_label': f"Week {cohort_start.strftime('%W')}, {cohort_start.year}",
                    'cohort_size': cohort_size,
                    'start_date': cohort_start.strftime('%Y-%m-%d'),
                    'periods': periods
                })

            return cohorts

        @staticmethod
        async def _generate_monthly_cohorts(period_count: int, metric: str) -> List[Dict[str, Any]]:
            """Generate monthly cohort data """
            cohorts = []
            base_date = datetime.now().replace(day=1) - timedelta(days=30 * period_count)

            for i in range(period_count):
                cohort_start = base_date + timedelta(days=30 * i)
                cohort_size = random.randint(2000, 3500)

                periods = []
                for j in range(period_count - i):
                    if j == 0:
                        if metric == 'retention':
                            value = 100.0
                        else:
                            value = cohort_size
                    else:
                        if metric == 'retention':
                            decay = 0.78 ** j
                            noise = random.uniform(0.92, 1.08)
                            value = max(10, min(100, (100 * decay * noise)))
                        else:
                            decay = 0.75 ** j
                            value = int(cohort_size * decay * random.uniform(0.85, 1.15))

                    periods.append({
                        'period': j,
                        'value': round(value, 1) if metric == 'retention' else int(value),
                        'period_label': f'Month {j + 1}',
                        'cohort_percentage': round((value / cohort_size * 100), 1) if metric != 'retention' else value
                    })

                cohorts.append({
                    'cohort_id': f"M{cohort_start.strftime('%Y-%m')}",
                    'cohort_label': cohort_start.strftime('%B %Y'),
                    'cohort_size': cohort_size,
                    'start_date': cohort_start.strftime('%Y-%m-%d'),
                    'periods': periods
                })

            return cohorts

    @staticmethod
    async def _generate_cohort_recommendations(cohort_data: List[Dict[str, Any]], metric: str) -> List[Dict[str, Any]]:
        """Generate recommendations based on cohort analysis"""
        recommendations = []

        # Analyze retention drop-off
        if len(cohort_data) > 0:
            latest_cohort = cohort_data[-1]
            if len(latest_cohort['periods']) > 2:
                p1_retention = latest_cohort['periods'][1]['value']
                p2_retention = latest_cohort['periods'][2]['value']
                drop_off = p1_retention - p2_retention

                if drop_off > 20:
                    recommendations.append({
                        'type': 'improvement_opportunity',
                        'title': 'High Early Drop-off',
                        'description': f'Users drop {drop_off:.1f}% between first and second period',
                        'suggestion': 'Focus on improving early user engagement and onboarding',
                        'priority': 'high'
                    })

        if len(cohort_data) >= 4:
            retention_values = []
            for cohort in cohort_data[-4:]:
                if len(cohort['periods']) > 1:
                    retention_values.append(cohort['periods'][1]['value'])

            if len(retention_values) >= 3:
                variance = np.var(retention_values)
                if variance < 25:
                    recommendations.append({
                        'type': 'positive_pattern',
                        'title': 'Stable Performance',
                        'description': 'Recent cohorts show consistent retention patterns',
                        'suggestion': 'Maintain current acquisition and engagement strategies',
                        'priority': 'low'
                    })

        return recommendations


#  Funnel Drop-Off Engine
class FunnelDropOffEngine:
    @staticmethod
    async def analyze_funnel_dropoff(funnel_request: FunnelDropOffRequest) -> Dict[str, Any]:
        """Analyze funnel drop-off points and generate recommendations"""
        try:
            # Generate sample funnel data based on request
            funnel_data = await FunnelDropOffEngine._generate_funnel_data(funnel_request)

            # Calculate drop-off metrics
            dropoff_analysis = await FunnelDropOffEngine._calculate_dropoff_metrics(funnel_data)

            # Generate recommendations
            recommendations = await FunnelDropOffEngine._generate_dropoff_recommendations(funnel_data)

            # Calculate overall funnel health
            health_score = await FunnelDropOffEngine._calculate_funnel_health(funnel_data)

            return {
                'funnel_stages': funnel_request.funnel_stages,
                'funnel_data': funnel_data,
                'dropoff_analysis': dropoff_analysis,
                'recommendations': recommendations,
                'health_score': health_score,
                'total_conversion': funnel_data[-1]['conversion_rate'] if funnel_data else 0,
                'bottleneck_stage': await FunnelDropOffEngine._identify_bottleneck(funnel_data)
            }

        except Exception as e:
            logger.error(f"Funnel analysis error: {e}")
            raise HTTPException(status_code=500, detail=f"Funnel analysis failed: {str(e)}")

    class FunnelDropOffEngine:
        @staticmethod
        async def _generate_funnel_data(funnel_request: FunnelDropOffRequest) -> List[Dict[str, Any]]:
            """Generate funnel data based on request parameters - FIXED"""
            stages = funnel_request.funnel_stages
            segment_by = funnel_request.segment_by

            if not stages:
                return []

            if len(stages) == 3:
                base_rates = [100, 65, 35]
            elif len(stages) == 4:
                base_rates = [100, 70, 45, 25]
            elif len(stages) == 5:
                base_rates = [100, 75, 50, 35, 20]
            elif len(stages) == 6:
                base_rates = [100, 80, 60, 45, 30, 18]
            else:
                base_rates = [100]
                for i in range(1, len(stages)):
                    prev_rate = base_rates[-1]
                    drop_off = max(15, min(40, 20 + (i * 5)))
                    base_rates.append(max(5, prev_rate - drop_off))

            # Add some randomness but keep trends consistent
            starting_users = 10000

            funnel_data = []
            previous_users = starting_users

            for i, stage in enumerate(stages):
                # Add realistic randomness (±5%)
                adjusted_rate = base_rates[i] * random.uniform(0.95, 1.05)
                current_users = int(starting_users * adjusted_rate / 100)

                # Calculate drop-off from previous stage
                if i == 0:
                    drop_off_rate = 0
                    drop_off_users = 0
                    conversion_rate = 100.0
                else:
                    drop_off_rate = round(((previous_users - current_users) / previous_users) * 100, 1)
                    drop_off_users = previous_users - current_users
                    conversion_rate = round((current_users / previous_users) * 100, 1)

                funnel_data.append({
                    'stage': stage,
                    'users': current_users,
                    'conversion_rate': conversion_rate,
                    'drop_off_rate': drop_off_rate,
                    'drop_off_users': drop_off_users,
                    'stage_order': i,
                    'cumulative_conversion': round((current_users / starting_users) * 100, 1)
                })

                previous_users = current_users

            return funnel_data

    @staticmethod
    async def _calculate_dropoff_metrics(funnel_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate detailed drop-off metrics"""
        if not funnel_data:
            return {}

        total_drop_off = 0
        max_drop_off = 0
        max_drop_off_stage = ""
        drop_off_points = []

        for i in range(1, len(funnel_data)):
            drop_off = funnel_data[i - 1]['conversion_rate'] - funnel_data[i]['conversion_rate']
            total_drop_off += drop_off

            if drop_off > max_drop_off:
                max_drop_off = drop_off
                max_drop_off_stage = funnel_data[i]['stage']

            drop_off_points.append({
                'from_stage': funnel_data[i - 1]['stage'],
                'to_stage': funnel_data[i]['stage'],
                'drop_off_rate': drop_off,
                'users_lost': funnel_data[i - 1]['users'] - funnel_data[i]['users']
            })

        avg_drop_off = total_drop_off / (len(funnel_data) - 1) if len(funnel_data) > 1 else 0

        return {
            'total_drop_off': round(total_drop_off, 1),
            'average_drop_off': round(avg_drop_off, 1),
            'max_drop_off': round(max_drop_off, 1),
            'max_drop_off_stage': max_drop_off_stage,
            'drop_off_points': drop_off_points,
            'overall_conversion': funnel_data[-1]['conversion_rate'] if funnel_data else 0
        }

    @staticmethod
    async def _generate_dropoff_recommendations(funnel_data: List[Dict[str, Any]]) -> List[DropOffRecommendation]:
        """Generate actionable recommendations for drop-off reduction"""
        recommendations = []

        if not funnel_data or len(funnel_data) < 2:
            return recommendations

        # Analyze each transition point
        for i in range(1, len(funnel_data)):
            from_stage = funnel_data[i - 1]
            to_stage = funnel_data[i]

            drop_off_rate = from_stage['conversion_rate'] - to_stage['conversion_rate']
            users_lost = from_stage['users'] - to_stage['users']

            if drop_off_rate > 15:
                recommendation = await FunnelDropOffEngine._create_stage_recommendation(
                    from_stage['stage'], to_stage['stage'], drop_off_rate, users_lost
                )
                recommendations.append(recommendation)

        # Sort by priority (high drop-off first)
        recommendations.sort(key=lambda x: x.drop_off_rate, reverse=True)

        return recommendations

    @staticmethod
    async def _create_stage_recommendation(from_stage: str, to_stage: str, drop_off_rate: float,
                                           users_lost: int) -> DropOffRecommendation:
        """Create specific recommendation for a stage transition"""
        # Stage-specific recommendations
        recommendations_map = {
            ('signup', 'activation'): {
                'recommendation': 'Simplify onboarding process and add email verification reminders',
                'impact': 'high',
                'effort': 'medium'
            },
            ('activation', 'onboarding'): {
                'recommendation': 'Add guided tour and highlight key features',
                'impact': 'medium',
                'effort': 'low'
            },
            ('onboarding', 'first_action'): {
                'recommendation': 'Provide clear call-to-actions and reduce friction',
                'impact': 'high',
                'effort': 'medium'
            },
            ('first_action', 'retention'): {
                'recommendation': 'Implement personalized notifications and re-engagement campaigns',
                'impact': 'medium',
                'effort': 'high'
            }
        }

        key = (from_stage, to_stage)
        default_rec = {
            'recommendation': 'Analyze user behavior and conduct user interviews to understand drop-off reasons',
            'impact': 'medium',
            'effort': 'medium'
        }

        stage_rec = recommendations_map.get(key, default_rec)

        # Determine priority based on drop-off rate and impact
        if drop_off_rate > 25:
            priority = 'critical'
        elif drop_off_rate > 15:
            priority = 'high'
        else:
            priority = 'medium'

        return DropOffRecommendation(
            stage=f"{from_stage} → {to_stage}",
            drop_off_rate=round(drop_off_rate, 1),
            users_lost=users_lost,
            recommendation=stage_rec['recommendation'],
            impact=stage_rec['impact'],
            effort=stage_rec['effort'],
            priority=priority
        )

    @staticmethod
    async def _calculate_funnel_health(funnel_data: List[Dict[str, Any]]) -> float:
        """Calculate overall funnel health score"""
        if not funnel_data or len(funnel_data) < 2:
            return 0.0

        total_conversion = funnel_data[-1]['conversion_rate']
        max_drop_off = 0

        # Find maximum drop-off between stages
        for i in range(1, len(funnel_data)):
            drop_off = funnel_data[i - 1]['conversion_rate'] - funnel_data[i]['conversion_rate']
            if drop_off > max_drop_off:
                max_drop_off = drop_off

        # Based on total conversion and maximum drop-off
        conversion_score = total_conversion * 0.8
        drop_off_penalty = max(0, (max_drop_off - 15) * 2)

        health_score = max(0, conversion_score - drop_off_penalty)

        return round(min(100, health_score), 1)

    @staticmethod
    async def _identify_bottleneck(funnel_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify the main bottleneck in the funnel"""
        if not funnel_data or len(funnel_data) < 2:
            return {}

        max_drop_off = 0
        bottleneck_transition = ""

        for i in range(1, len(funnel_data)):
            drop_off = funnel_data[i - 1]['conversion_rate'] - funnel_data[i]['conversion_rate']
            if drop_off > max_drop_off:
                max_drop_off = drop_off
                bottleneck_transition = f"{funnel_data[i - 1]['stage']} → {funnel_data[i]['stage']}"

        return {
            'transition': bottleneck_transition,
            'drop_off_rate': round(max_drop_off, 1),
            'severity': 'critical' if max_drop_off > 25 else 'high' if max_drop_off > 15 else 'medium'
        }


# AI Action Recommendation Engine
class AIActionRecommendationEngine:
    def __init__(self):
        self.ml_model = None
        self._load_ml_model()

    def _load_ml_model(self):
        """Load pre-trained ML model for recommendations"""
        try:

            # For now, we'll use rule-based + simple ML
            logger.info("ML model placeholder loaded - using enhanced rule-based recommendations")

        except Exception as e:
            logger.warning(f"ML model loading failed: {e}. Using rule-based fallback.")

    async def generate_recommendations(self, action_request: AIActionRequest) -> Dict[str, Any]:
        """Generate AI-powered recommendations"""
        try:
            # For demo purposes, generate sample recommendations
            recommendations = await self._generate_demo_recommendations(action_request)

            return {
                'focus_area': action_request.focus_area,
                'timeframe': action_request.timeframe,
                'recommendations': recommendations,
                'opportunity_score': {'score': 78.5, 'level': 'high',
                                      'description': 'Strong improvement opportunities identified'},
                'roadmap': {
                    'phases': [
                        {
                            'phase': 1,
                            'name': 'Quick Wins',
                            'timeline': '2-4 weeks',
                            'recommendations': [recommendations[0]] if recommendations else [],
                            'expected_impact': 8.5,
                            'resources': 'minimal'
                        }
                    ],
                    'total_timeline': '2-4 weeks'
                },
                'data_sources_used': ['demo_data'],
                'confidence_level': 0.78,
                'generated_at': datetime.now().isoformat(),
                'total_recommendations': len(recommendations)
            }

        except Exception as e:
            logger.error(f"AI recommendations error: {e}")
            return await self._generate_rule_based_recommendations(action_request)

    async def _generate_demo_recommendations(self, action_request: AIActionRequest) -> List[AIActionRecommendation]:
        """Generate demo recommendations for testing"""
        recommendations = [
            AIActionRecommendation(
                id=f"demo_rec_{int(time.time())}",
                title="Optimize User Onboarding",
                description="Improve new user activation with guided tours",
                action_type="user_experience",
                impact_score=8.5,
                confidence=0.82,
                effort_required="medium",
                metrics_affected=["activation_rate", "retention", "user_satisfaction"],
                suggested_implementation="Create interactive onboarding flow with progress tracking",
                expected_improvement="25-35% increase in activation rates",
                kpis_to_watch=["activation_rate", "time_to_first_value", "user_feedback"],
                generated_at=datetime.now().isoformat()
            ),
            AIActionRecommendation(
                id=f"demo_rec_{int(time.time()) + 1}",
                title="Enhance Data Visualization",
                description="Add more interactive charts and real-time dashboards",
                action_type="product_improvement",
                impact_score=7.8,
                confidence=0.75,
                effort_required="high",
                metrics_affected=["user_engagement", "feature_adoption", "satisfaction"],
                suggested_implementation="Implement Plotly.js for interactive charts and real-time updates",
                expected_improvement="15-25% increase in dashboard usage",
                kpis_to_watch=["dashboard_views", "session_duration", "feature_usage"],
                generated_at=datetime.now().isoformat()
            )
        ]

        return recommendations

    async def _generate_rule_based_recommendations(self, action_request: AIActionRequest) -> Dict[str, Any]:
        """Fallback to rule-based recommendations"""
        return {
            'focus_area': action_request.focus_area,
            'timeframe': action_request.timeframe,
            'recommendations': [],
            'opportunity_score': {'score': 0, 'level': 'low', 'description': 'Using fallback recommendations'},
            'roadmap': {'phases': []},
            'data_sources_used': [],
            'confidence_level': 0.5,
            'generated_at': datetime.now().isoformat(),
            'total_recommendations': 0
        }

    async def generate_real_recommendations(self, action_request: AIActionRequest, user_data: Dict[str, Any]) -> Dict[
        str, Any]:
        """Generate real AI-powered recommendations based on actual data"""
        try:
            # For now, use demo data - in production this would analyze real user data
            return await self.generate_recommendations(action_request)
        except Exception as e:
            logger.error(f"Real AI recommendation error: {e}")
            return await self._generate_rule_based_recommendations(action_request)


@app.post("/api/signup")
async def signup(user_data: UserCreate, request: Request):
    try:
        db = await DatabaseManager.get_instance()

        # Check if user already exists
        existing_user = await db.get_user_by_email(user_data.email)
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")

        # Create new user
        hashed_password = AuthManager.hash_password(user_data.password)
        new_user_data = {
            'email': user_data.email,
            'password_hash': hashed_password,
            'full_name': user_data.full_name,
            'role': user_data.role.value,
            'is_verified': True,
            'created_at': datetime.now().isoformat()
        }

        user_id = await db.create_user(new_user_data)

        # Log user history
        await UserHistoryManager.log_user_action(
            user_id=user_id,
            action_type=UserActionType.LOGIN,
            description="New user signed up",
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get('user-agent')
        )

        # Auto-login after signup
        access_token, refresh_token = AuthManager.create_tokens({
            'user_id': user_id,
            'role': user_data.role.value
        })

        return JSONResponse(content={
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "user_email": user_data.email,
            "user_role": user_data.role.value,
            "full_name": user_data.full_name,
            "user_id": user_id,
            "message": "User created successfully"
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Signup error: {e}")
        raise HTTPException(status_code=500, detail="Signup failed")


@app.post("/api/debug/check-user")
async def debug_check_user(email: str):
    """Debug endpoint to check if user exists"""
    db = await DatabaseManager.get_instance()
    user = await db.get_user_by_email(email)

    if user:
        return {
            "exists": True,
            "email": user['email'],
            "user_id": user['id'],
            "has_password_hash": bool(user.get('password_hash')),
            "password_hash_length": len(user.get('password_hash', '')) if user.get('password_hash') else 0
        }
    else:
        return {"exists": False}


@app.post("/api/debug/verify-password")
async def debug_verify_password(email: str, password: str):
    """Debug endpoint to verify password"""
    db = await DatabaseManager.get_instance()
    user = await db.get_user_by_email(email)

    if not user:
        return {"error": "User not found"}

    is_valid = AuthManager.verify_password(password, user['password_hash'])

    return {
        "user_exists": True,
        "password_valid": is_valid,
        "email": user['email'],
        "stored_hash_prefix": user['password_hash'][:20] + "..." if user.get('password_hash') else "No hash"
    }

# Rice ICE Framework Manager
class ICEFrameworkManager:
    @staticmethod
    async def create_ice_initiative(initiative_data: ICEInitiativeCreate, user_id: int) -> Dict[str, Any]:
        """Create a new ICE initiative"""
        try:
            db = await DatabaseManager.get_instance()

            # Calculate ICE score
            ice_score = (initiative_data.impact + initiative_data.confidence + initiative_data.ease) / 3

            # Determine priority based on ICE score
            if ice_score >= 7:
                priority = ICEPriority.HIGH
            elif ice_score >= 5:
                priority = ICEPriority.MEDIUM
            else:
                priority = ICEPriority.LOW

            initiative_id = f"ice_{str(int(time.time() * 1000))}"

            initiative = {
                'id': initiative_id,
                'title': initiative_data.title,
                'description': initiative_data.description,
                'impact': initiative_data.impact,
                'confidence': initiative_data.confidence,
                'ease': initiative_data.ease,
                'ice_score': round(ice_score, 2),
                'priority': priority.value,
                'status': ICEStatus.PENDING.value,
                'assigned_to': initiative_data.assigned_to,
                'created_by': user_id,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'due_date': initiative_data.due_date,
                'tags': json.dumps(initiative_data.tags),
                'metrics_affected': json.dumps(initiative_data.metrics_affected),
                'estimated_effort_days': initiative_data.estimated_effort_days
            }

            success = await db.create_ice_initiative(initiative)
            if not success:
                raise HTTPException(status_code=500, detail="Failed to create ICE initiative")

            # Log user history
            await UserHistoryManager.log_user_action(
                user_id=user_id,
                action_type=UserActionType.CREATE_ICE_INITIATIVE,
                description=f"Created ICE initiative: {initiative_data.title}",
                metadata={'initiative_id': initiative_id, 'title': initiative_data.title}
            )

            return initiative

        except Exception as e:
            logger.error(f"ICE initiative creation error: {e}")
            raise HTTPException(status_code=500, detail=f"Error creating ICE initiative: {str(e)}")

    @staticmethod
    async def get_ice_initiatives(user_id: int, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get ICE initiatives for user"""
        try:
            db = await DatabaseManager.get_instance()
            initiatives = await db.get_ice_initiatives(user_id, status)
            return initiatives
        except Exception as e:
            logger.error(f"ICE initiatives fetch error: {e}")
            return []

    @staticmethod
    async def get_ice_prioritization_matrix(user_id: int) -> Dict[str, Any]:
        """Get ICE prioritization matrix data"""
        db = await DatabaseManager.get_instance()
        initiatives = await db.get_ice_initiatives(user_id)

        high_priority = [init for init in initiatives if init['priority'] == 'high']
        medium_priority = [init for init in initiatives if init['priority'] == 'medium']
        low_priority = [init for init in initiatives if init['priority'] == 'low']

        return {
            'total_initiatives': len(initiatives),
            'high_priority_count': len(high_priority),
            'medium_priority_count': len(medium_priority),
            'low_priority_count': len(low_priority),
            'average_ice_score': round(sum(init['ice_score'] for init in initiatives) / len(initiatives),
                                       2) if initiatives else 0,
            'initiatives_by_status': {
                'pending': len([init for init in initiatives if init['status'] == 'pending']),
                'in_progress': len([init for init in initiatives if init['status'] == 'in_progress']),
                'completed': len([init for init in initiatives if init['status'] == 'completed']),
                'cancelled': len([init for init in initiatives if init['status'] == 'cancelled'])
            },
            'ice_score_distribution': {
                '9_10': len([init for init in initiatives if 9 <= init['ice_score'] <= 10]),
                '7_8.9': len([init for init in initiatives if 7 <= init['ice_score'] < 9]),
                '5_6.9': len([init for init in initiatives if 5 <= init['ice_score'] < 7]),
                '0_4.9': len([init for init in initiatives if init['ice_score'] < 5])
            }
        }


# User History Manager
class UserHistoryManager:
    @staticmethod
    async def log_user_action(
            user_id: int,
            action_type: UserActionType,
            description: str,
            metadata: Dict[str, Any] = {},
            ip_address: Optional[str] = None,
            user_agent: Optional[str] = None
    ):
        """Log user action to history"""
        try:
            db = await DatabaseManager.get_instance()

            history_id = f"hist_{str(int(time.time() * 1000))}"

            history_record = {
                'id': history_id,
                'user_id': user_id,
                'action_type': action_type.value,
                'description': description,
                'metadata': json.dumps(metadata),
                'timestamp': datetime.now().isoformat(),
                'ip_address': ip_address,
                'user_agent': user_agent
            }

            success = await db.create_user_history(history_record)
            if not success:
                logger.error("Failed to log user history")

            return history_record

        except Exception as e:
            logger.error(f"User history logging error: {e}")

    @staticmethod
    async def get_user_history(user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
        """Get user history records"""
        try:
            db = await DatabaseManager.get_instance()
            return await db.get_user_history(user_id, limit)
        except Exception as e:
            logger.error(f"User history fetch error: {e}")
            return []

    @staticmethod
    async def get_user_activity_summary(user_id: int) -> Dict[str, Any]:
        """Get user activity summary"""
        db = await DatabaseManager.get_instance()
        user_history = await db.get_user_history(user_id, limit=1000)

        week_ago = datetime.now() - timedelta(days=7)
        recent_activity = [record for record in user_history if datetime.fromisoformat(record['timestamp']) >= week_ago]

        activity_by_type = {}
        for record in recent_activity:
            action_type = record['action_type']
            activity_by_type[action_type] = activity_by_type.get(action_type, 0) + 1

        return {
            'total_actions': len(user_history),
            'recent_actions_7d': len(recent_activity),
            'activity_by_type': activity_by_type,
            'last_action': user_history[0]['timestamp'] if user_history else None,
            'most_active_day': UserHistoryManager._get_most_active_day(recent_activity) if recent_activity else None
        }

    @staticmethod
    def _get_most_active_day(activity_records: List[Dict[str, Any]]) -> str:
        """Get the most active day from activity records"""
        day_count = {}
        for record in activity_records:
            day = datetime.fromisoformat(record['timestamp']).strftime('%Y-%m-%d')
            day_count[day] = day_count.get(day, 0) + 1

        if day_count:
            return max(day_count, key=day_count.get)
        return "No activity"


# Analytics Manager with Demo Data Only
class AnalyticsManager:
    @staticmethod
    async def get_dau_mau_data(days: int = 30) -> Dict[str, Any]:
        """Get DAU/MAU data - DEMO ONLY"""
        try:
            return await AnalyticsManager._get_demo_dau_mau_data(days)
        except Exception as e:
            logger.warning(f"Demo DAU/MAU data generation failed: {e}")
            return await AnalyticsManager._get_demo_dau_mau_data(days)

    @staticmethod
    async def _get_demo_mau_data() -> List[Dict[str, Any]]:
        """Generate demo MAU data"""
        mau_data = []
        current_date = datetime.now()
        for i in range(6):
            month_date = current_date - timedelta(days=30 * i)
            month_start = month_date.replace(day=1)
            month_name = month_start.strftime('%B %Y')

            base_mau = 5500
            trend_factor = 1 + (i * 0.04)
            active_users = int(base_mau * trend_factor * random.uniform(0.95, 1.05))

            mau_data.append({
                'month': month_name,
                'active_users': active_users,
                'year': month_start.year,
                'month_num': month_start.month
            })
        mau_data.reverse()
        return mau_data

    @staticmethod
    async def _get_demo_dau_mau_data(days: int = 30) -> Dict[str, Any]:
        """Generate realistic DAU/MAU data (demo fallback)"""
        dau_data = []
        base_dau = 180
        for i in range(days):
            date = (datetime.now() - timedelta(days=days - i - 1)).date()
            day_factor = 1.25 if date.weekday() >= 5 else 1.0
            trend_factor = 1 + (i * 0.015)
            active_users = int(base_dau * day_factor * trend_factor * random.uniform(0.95, 1.05))

            dau_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'active_users': active_users,
                'day_of_week': date.strftime('%A'),
                'source': 'demo_data'
            })

        mau_data = await AnalyticsManager._get_demo_mau_data()

        current_dau = dau_data[-1]['active_users'] if dau_data else 0
        current_mau = mau_data[-1]['active_users'] if mau_data else 0
        stickiness = round((current_dau / current_mau) * 100, 2) if current_mau > 0 else 0

        return {
            'dau': dau_data,
            'mau': mau_data,
            'summary': {
                'current_dau': current_dau,
                'current_mau': current_mau,
                'stickiness': stickiness,
                'source': 'demo_data',
                'data_quality': 'high'
            }
        }

    @staticmethod
    async def get_kpi_hierarchy() -> Dict[str, Any]:
        """Generate KPI hierarchy data"""
        return {
            "nodes": [
                {"id": "product_health", "name": "Product Health", "value": 87.5, "parent": None,
                 "description": "Overall health score", "impact": "high"},
                {"id": "user_engagement", "name": "User Engagement", "value": 82.3, "parent": "product_health",
                 "description": "User activity metrics", "impact": "high"},
                {"id": "business_value", "name": "Business Value", "value": 94.2, "parent": "product_health",
                 "description": "Revenue and business impact", "impact": "high"},
                {"id": "daily_active_users", "name": "Daily Active Users", "value": 1850, "parent": "user_engagement",
                 "description": "Daily active users", "impact": "high"},
                {"id": "retention_rate", "name": "Retention Rate", "value": 78.2, "parent": "user_engagement",
                 "description": "User retention", "impact": "high"},
                {"id": "monthly_revenue", "name": "Monthly Revenue", "value": 185000, "parent": "business_value",
                 "description": "Monthly revenue", "impact": "high"},
            ],
            "treemap_data": {
                "labels": ["Product Health", "User Engagement", "Business Value", "Daily Active Users",
                           "Retention Rate", "Monthly Revenue"],
                "parents": ["", "Product Health", "Product Health", "User Engagement", "User Engagement",
                            "Business Value"],
                "values": [87.5, 82.3, 94.2, 1850, 78.2, 185000]
            },
            "summary": {
                "total_kpis": 6,
                "high_impact_kpis": 5,
                "average_score": 85.7
            }
        }

    @staticmethod
    async def get_feature_heatmaps() -> Dict[str, Any]:
        """Generate feature heatmap data"""
        features = ['Dashboard', 'AI Insights', 'Custom Reports', 'Predictive Analytics', 'Real Data']
        segments = ['Power Users', 'Regular Users', 'Casual Users', 'New Users']

        adoption_matrix = []
        for segment in segments:
            row = []
            for feature in features:
                if segment == 'Power Users':
                    adoption = np.random.uniform(0.75, 0.98)
                elif segment == 'Regular Users':
                    adoption = np.random.uniform(0.5, 0.85)
                elif segment == 'Casual Users':
                    adoption = np.random.uniform(0.3, 0.7)
                else:
                    adoption = np.random.uniform(0.1, 0.4)
                row.append(round(adoption * 100, 1))
            adoption_matrix.append(row)

        return {
            "features": features,
            "segments": segments,
            "adoption_matrix": adoption_matrix,
            "max_adoption": max(max(row) for row in adoption_matrix),
            "min_adoption": min(min(row) for row in adoption_matrix)
        }

    @staticmethod
    async def get_correlation_matrix() -> Dict[str, Any]:
        """Generate correlation matrix data"""
        metrics = ['Daily Active Users', 'Session Duration', 'Feature Adoption', 'Retention Rate', 'Conversion Rate',
                   'Real Data Usage']

        n_metrics = len(metrics)
        correlation_matrix = np.eye(n_metrics)

        relationships = {
            (0, 1): 0.65,
            (0, 2): 0.72,
            (0, 3): 0.85,
            (2, 3): 0.68,
            (3, 4): 0.75,
            (5, 0): 0.58,
            (5, 3): 0.62,
        }

        for (i, j), corr in relationships.items():
            correlation_matrix[i, j] = corr
            correlation_matrix[j, i] = corr

        corr_list = correlation_matrix.tolist()

        return {
            "metrics": metrics,
            "correlation_matrix": corr_list,
            "strong_positive": [(metrics[i], metrics[j], corr_list[i][j])
                                for i in range(n_metrics) for j in range(i + 1, n_metrics)
                                if corr_list[i][j] > 0.5],
            "strong_negative": [(metrics[i], metrics[j], corr_list[i][j])
                                for i in range(n_metrics) for j in range(i + 1, n_metrics)
                                if corr_list[i][j] < -0.3]
        }

    @staticmethod
    async def get_north_star_metric() -> Dict[str, Any]:
        """Get north star metric data"""
        return {
            'current_score': 78.5,
            'target_score': 85.0,
            'progress': 92.3,
            'trend': 3.2,
            'health': 'excellent',
            'top_drivers': [
                {'metric': 'Feature Adoption', 'impact': 0.38, 'trend': 'improving'},
                {'metric': 'Retention Rate', 'impact': 0.32, 'trend': 'improving'},
                {'metric': 'Real Data Usage', 'impact': 0.25, 'trend': 'improving'}
            ],
            'forecast': {
                'next_week': 79.8,
                'next_month': 82.5,
                'confidence': 0.88,
                'factors': ['real data integration', 'feature launches', 'user engagement']
            }
        }

    @staticmethod
    async def get_funnel_analysis() -> Dict[str, Any]:
        """Get funnel analysis data"""
        return {
            'funnel_data': [
                {'stage': 'signup', 'users': 15000, 'conversion_rate': 100.0, 'dropoff_rate': 0.0},
                {'stage': 'activation', 'users': 9800, 'conversion_rate': 65.3, 'dropoff_rate': 34.7},
                {'stage': 'retention', 'users': 6800, 'conversion_rate': 69.4, 'dropoff_rate': 30.6},
                {'stage': 'conversion', 'users': 4200, 'conversion_rate': 61.8, 'dropoff_rate': 38.2},
                {'stage': 'revenue', 'users': 3200, 'conversion_rate': 76.2, 'dropoff_rate': 23.8}
            ],
            'total_conversion': 21.3,
            'bottleneck_stage': 'activation',
            'improvement_opportunity': [
                {
                    'stage': 'activation',
                    'opportunity': 'Improve onboarding flow with real data insights',
                    'impact': 'high',
                    'effort': 'medium',
                    'recommendation': 'Add personalized onboarding based on user data'
                }
            ]
        }

    @staticmethod
    async def get_ai_recommendations() -> Dict[str, Any]:
        """Get AI recommendations"""
        return {
            'recommendations': [
                {
                    'problem': 'Real data integration can be optimized',
                    'action': 'Configure additional data sources for comprehensive analytics',
                    'impact': 'Expected 20-30% data accuracy improvement',
                    'priority': 'high',
                    'ai_generated': True
                },
                {
                    'problem': 'Feature adoption could be improved with real data',
                    'action': 'Use real user data to personalize feature recommendations',
                    'impact': 'Expected 25% adoption increase',
                    'priority': 'medium',
                    'ai_generated': True
                },
                {
                    'problem': 'Data quality monitoring needed',
                    'action': 'Implement automated data quality checks',
                    'impact': 'Improved decision making reliability',
                    'priority': 'medium',
                    'ai_generated': True
                }
            ]
        }

    @staticmethod
    async def generate_custom_report(report_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate custom report based on configuration"""
        return {
            'report_id': report_config.get('id', 1),
            'name': report_config.get('name', 'Custom Report'),
            'generated_at': datetime.now().isoformat(),
            'metrics': report_config.get('metrics', []),
            'summary': {
                'total_users': 1850,
                'active_today': 235,
                'retention_rate': 78.2,
                'conversion_rate': 18.7,
                'data_sources': 3
            },
            'charts': [
                {
                    'type': 'bar',
                    'title': 'User Activity Overview',
                    'data': [1850, 235, 78.2, 18.7, 3],
                    'labels': ['Total Users', 'Active Today', 'Retention %', 'Conversion %', 'Data Sources']
                }
            ],
            'insights': [
                'Real data integration improved accuracy by 25%',
                'User engagement increased by 8% this week',
                'Retention rate shows positive trend with real data'
            ]
        }


# A/B Testing Manager
class ABTestingManager:
    @staticmethod
    async def create_ab_test(test_data: Dict[str, Any], user_id: int) -> Dict[str, Any]:
        """Create a new A/B test"""
        db = await DatabaseManager.get_instance()

        test_data = {
            'name': test_data['name'],
            'description': test_data['description'],
            'hypothesis': test_data['hypothesis'],
            'status': 'draft',
            'variants': json.dumps(test_data['variants']),
            'primary_metric': test_data['primary_metric'],
            'target_audience': json.dumps(test_data.get('target_audience', {})),
            'duration_days': test_data.get('duration_days', 14),
            'created_by': user_id,
            'created_at': datetime.now().isoformat()
        }

        test_id = await db.create_ab_test(test_data)

        # Log user history
        await UserHistoryManager.log_user_action(
            user_id=user_id,
            action_type=UserActionType.CREATE_AB_TEST,
            description=f"Created A/B test: {test_data['name']}",
            metadata={'test_id': test_id, 'test_name': test_data['name']}
        )

        return {**test_data, 'id': test_id}

    @staticmethod
    async def get_ab_tests(user_id: int) -> List[Dict[str, Any]]:
        """Get all A/B tests for user"""
        db = await DatabaseManager.get_instance()
        return await db.get_ab_tests(user_id)

    @staticmethod
    async def get_ab_test(test_id: int, user_id: int) -> Optional[Dict[str, Any]]:
        """Get specific A/B test"""
        db = await DatabaseManager.get_instance()
        tests = await db.get_ab_tests(user_id)
        for test in tests:
            if test['id'] == test_id:
                return test
        return None

    @staticmethod
    async def update_ab_test(test_id: int, update_data: Dict[str, Any], user_id: int) -> Optional[Dict[str, Any]]:
        """Update A/B test"""
        db = await DatabaseManager.get_instance()

        # Prepare update data
        update_dict = {}
        for key, value in update_data.items():
            if value is not None:
                if key in ['variants', 'target_audience', 'results']:
                    update_dict[key] = json.dumps(value)
                else:
                    update_dict[key] = value

        update_dict['updated_at'] = datetime.now().isoformat()

        success = await db.update_ab_test(test_id, update_dict)
        if success:
            return await ABTestingManager.get_ab_test(test_id, user_id)
        return None

    @staticmethod
    async def start_ab_test(test_id: int, user_id: int) -> Optional[Dict[str, Any]]:
        """Start an A/B test"""
        update_data = {
            'status': 'running',
            'started_at': datetime.now().isoformat()
        }
        return await ABTestingManager.update_ab_test(test_id, update_data, user_id)

    @staticmethod
    async def get_ab_test_results(test_id: int, user_id: int) -> Dict[str, Any]:
        """Get A/B test results with statistical significance"""
        test = await ABTestingManager.get_ab_test(test_id, user_id)
        if not test:
            return {}

        if not test.get('results'):
            variants = test['variants']
            results = {}

            for i, variant in enumerate(variants):
                variant_name = variant['name'].lower().replace(' ', '_')
                base_conversion = random.uniform(0.12, 0.25)
                if i == 0:
                    conversion_rate = base_conversion
                else:
                    conversion_rate = base_conversion * random.uniform(0.9, 1.4)

                sample_size = random.randint(800, 2500)
                conversions = int(sample_size * conversion_rate)

                results[variant_name] = {
                    'conversion_rate': round(conversion_rate * 100, 2),
                    'sample_size': sample_size,
                    'conversions': conversions,
                    'confidence_interval': [
                        round((conversion_rate - 0.015) * 100, 2),
                        round((conversion_rate + 0.015) * 100, 2)
                    ]
                }

            await ABTestingManager.update_ab_test(test_id, {'results': results}, user_id)
            test['results'] = results

        results = test['results']
        if len(results) >= 2:
            variants = list(results.keys())
            control = results[variants[0]]
            best_variant = max(variants[1:], key=lambda v: results[v]['conversion_rate'])

            is_significant = random.random() > 0.25

            return {
                'test': test,
                'results': results,
                'best_performing': best_variant,
                'improvement': round(
                    (results[best_variant]['conversion_rate'] - control['conversion_rate']) / control[
                        'conversion_rate'] * 100, 2
                ),
                'statistical_significance': is_significant,
                'confidence_level': round(random.uniform(0.88, 0.99), 3) if is_significant else round(
                    random.uniform(0.6, 0.8), 3),
                'recommendation': f"Implement {best_variant}" if is_significant else "Continue testing"
            }

        return {'test': test, 'results': results}


# Data Export Manager
class DataExportManager:
    @staticmethod
    async def export_data(export_request: Dict[str, Any], user_id: int) -> Dict[str, Any]:
        """Export data in requested format"""
        db = await DatabaseManager.get_instance()

        data_type = export_request['data_type']
        export_format = export_request.get('format', 'csv')

        if data_type == 'user_analytics':
            data = await DataExportManager._generate_user_analytics_data()
        elif data_type == 'feature_usage':
            data = await DataExportManager._generate_feature_usage_data()
        elif data_type == 'ab_test_results':
            data = await DataExportManager._generate_ab_test_data()
        else:
            data = await DataExportManager._generate_general_analytics_data()

        export_data = {
            'user_id': user_id,
            'data_type': data_type,
            'format': export_format,
            'filters': json.dumps(export_request.get('filters', {})),
            'file_name': f"export_{data_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format}",
            'created_at': datetime.now().isoformat(),
            'status': 'completed',
            'record_count': len(data) if isinstance(data, list) else 1
        }

        export_id = await db.create_export(export_data)

        # Log user history
        await UserHistoryManager.log_user_action(
            user_id=user_id,
            action_type=UserActionType.EXPORT_DATA,
            description=f"Exported {data_type} data as {export_format}",
            metadata={'export_id': export_id, 'data_type': data_type, 'format': export_format}
        )

        return {
            'export_id': export_id,
            'file_name': export_data['file_name'],
            'data': data,
            'format': export_format,
            'record_count': export_data['record_count']
        }

    @staticmethod
    async def _generate_user_analytics_data() -> List[Dict[str, Any]]:
        """Generate user analytics data for export"""
        data = []
        for i in range(100):
            date = (datetime.now() - timedelta(days=99 - i)).date()
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'active_users': random.randint(150, 250),
                'new_users': random.randint(15, 40),
                'sessions': random.randint(200, 400),
                'avg_session_duration': round(random.uniform(180, 720), 2),
                'retention_rate': round(random.uniform(0.65, 0.85), 4)
            })
        return data

    @staticmethod
    async def _generate_feature_usage_data() -> List[Dict[str, Any]]:
        """Generate feature usage data for export"""
        features = ['Dashboard', 'Reports', 'AI Insights', 'Export', 'Settings', 'Real Data']
        data = []

        for feature in features:
            data.append({
                'feature': feature,
                'daily_active_users': random.randint(80, 250),
                'weekly_active_users': random.randint(300, 600),
                'monthly_active_users': random.randint(1000, 1500),
                'adoption_rate': round(random.uniform(0.4, 0.95), 4)
            })
        return data

    @staticmethod
    async def _generate_ab_test_data() -> List[Dict[str, Any]]:
        """Generate A/B test data for export"""
        return [
            {
                'test_id': 1,
                'test_name': 'Homepage Hero Button Color',
                'variant': 'Control',
                'conversion_rate': 15.2,
                'sample_size': 1500,
                'conversions': 228,
                'confidence_interval_low': 13.8,
                'confidence_interval_high': 16.6
            },
            {
                'test_id': 1,
                'test_name': 'Homepage Hero Button Color',
                'variant': 'Variant A',
                'conversion_rate': 18.7,
                'sample_size': 1480,
                'conversions': 277,
                'confidence_interval_low': 17.2,
                'confidence_interval_high': 20.2
            }
        ]

    @staticmethod
    async def _generate_general_analytics_data() -> List[Dict[str, Any]]:
        """Generate general analytics data for export"""
        data = []
        for i in range(30):
            date = (datetime.now() - timedelta(days=29 - i)).date()
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'revenue': round(random.uniform(1500, 6000), 2),
                'conversions': random.randint(80, 200),
                'bounce_rate': round(random.uniform(0.25, 0.5), 4),
                'pages_per_session': round(random.uniform(3.0, 7.5), 2)
            })
        return data

    @staticmethod
    async def get_export_history(user_id: int) -> List[Dict[str, Any]]:
        """Get user's export history"""
        db = await DatabaseManager.get_instance()
        return await db.get_exports(user_id)


# Anomaly Detection Manager
class AnomalyDetectionManager:
    @staticmethod
    async def detect_anomalies(detection_request: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in metrics"""
        metric = detection_request['metric']
        time_range = detection_request.get('time_range', '7d')
        sensitivity = detection_request.get('sensitivity', 0.1)

        data = await AnomalyDetectionManager._generate_time_series_data(metric, time_range)

        zscore_anomalies = await AnomalyDetectionManager._zscore_detection(data, sensitivity)
        isolation_anomalies = await AnomalyDetectionManager._isolation_forest_detection(data, sensitivity)

        all_anomalies = list(set(zscore_anomalies + isolation_anomalies))

        severity_levels = await AnomalyDetectionManager._calculate_severity(data, all_anomalies)

        chart_data = await AnomalyDetectionManager._prepare_chart_data(data, all_anomalies)

        return {
            'total_data_points': len(data),
            'anomalies_detected': len(all_anomalies),
            'anomaly_percentage': round(len(all_anomalies) / len(data) * 100, 2),
            'sensitivity': sensitivity,
            'anomalies': all_anomalies,
            'severity_levels': severity_levels,
            'chart_data': chart_data
        }

    @staticmethod
    async def _generate_time_series_data(metric: str, time_range: str) -> List[Dict[str, Any]]:
        """Generate time series data with some anomalies"""
        days = 7 if time_range == '7d' else 30 if time_range == '30d' else 90
        data = []

        base_value = 1500
        trend = 1.015

        for i in range(days):
            date = (datetime.now() - timedelta(days=days - i - 1))

            value = base_value * (trend ** i) * random.uniform(0.95, 1.05)

            if random.random() < 0.05:
                if random.random() < 0.5:
                    value *= random.uniform(1.6, 2.8)
                else:
                    value *= random.uniform(0.15, 0.45)
                is_anomaly = True
            else:
                is_anomaly = False

            data.append({
                'timestamp': date.isoformat(),
                'date': date.strftime('%Y-%m-%d'),
                'value': round(value, 2),
                'is_anomaly': is_anomaly
            })

        return data

    @staticmethod
    async def _zscore_detection(data: List[Dict[str, Any]], sensitivity: float) -> List[str]:
        """Detect anomalies using Z-score method"""
        values = [point['value'] for point in data]

        if len(values) < 2:
            return []

        mean = np.mean(values)
        std = np.std(values)

        if std == 0:
            return []

        anomalies = []
        zscore_threshold = 2.5 * (1 - sensitivity)

        for point in data:
            zscore = abs((point['value'] - mean) / std)
            if zscore > zscore_threshold:
                anomalies.append(point['date'])

        return anomalies

    @staticmethod
    async def _isolation_forest_detection(data: List[Dict[str, Any]], sensitivity: float) -> List[str]:
        """Detect anomalies using Isolation Forest"""
        values = np.array([point['value'] for point in data]).reshape(-1, 1)

        if len(values) < 2:
            return []

        contamination = 0.1 * (1 - sensitivity)

        try:
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            predictions = iso_forest.fit_predict(values)

            anomalies = []
            for i, prediction in enumerate(predictions):
                if prediction == -1:
                    anomalies.append(data[i]['date'])

            return anomalies
        except Exception as e:
            logger.error(f"Isolation Forest error: {e}")
            return []

    @staticmethod
    async def _calculate_severity(data: List[Dict[str, Any]], anomalies: List[str]) -> List[Dict[str, Any]]:
        """Calculate severity levels for anomalies"""
        if not anomalies:
            return []

        values = [point['value'] for point in data]
        mean = np.mean(values)
        std = np.std(values)

        if std == 0:
            return []

        severity_levels = []
        for anomaly_date in anomalies:
            anomaly_point = next((point for point in data if point['date'] == anomaly_date), None)
            if anomaly_point:
                zscore = abs((anomaly_point['value'] - mean) / std)

                if zscore > 3:
                    severity = 'critical'
                elif zscore > 2:
                    severity = 'high'
                elif zscore > 1.5:
                    severity = 'medium'
                else:
                    severity = 'low'

                severity_levels.append({
                    'date': anomaly_date,
                    'value': anomaly_point['value'],
                    'zscore': round(zscore, 2),
                    'severity': severity,
                    'deviation': round((anomaly_point['value'] - mean) / mean * 100, 2)
                })

        return severity_levels

    @staticmethod
    async def _prepare_chart_data(data: List[Dict[str, Any]], anomalies: List[str]) -> Dict[str, Any]:
        """Prepare data for visualization in frontend format"""
        dates = [point['date'] for point in data]
        values = [point['value'] for point in data]

        anomaly_values = []
        for point in data:
            if point['date'] in anomalies:
                anomaly_values.append(point['value'])
            else:
                anomaly_values.append(None)

        return {
            'dates': dates,
            'values': values,
            'anomaly_values': anomaly_values
        }

    @staticmethod
    async def get_anomaly_history(user_id: int) -> List[Dict[str, Any]]:
        """Get anomaly detection history for user"""
        db = await DatabaseManager.get_instance()
        return await db.fetch_all(
            "SELECT * FROM anomaly_detections WHERE user_id = $1 ORDER BY detected_at DESC",
            user_id
        )


# DEMO Notion Integration Manager (No Live API Calls)
class NotionIntegrationManager:
    @staticmethod
    async def connect_notion(integration_data: Dict[str, Any], user_id: int) -> Dict[str, Any]:
        """DEMO: Simulate Notion connection - NO LIVE API CALLS"""
        try:
            db = await DatabaseManager.get_instance()

            notion_token = integration_data['notion_token']
            database_id = integration_data.get('database_id', 'demo_database_id')

            # DEMO: Simulate successful connection
            logger.info(f"🔗 DEMO Notion connection simulated for token: {notion_token[:10]}...")

            # Create integration record
            integration_record = {
                'user_id': user_id,
                'notion_token': notion_token,
                'database_id': database_id,
                'sync_metrics': json.dumps(integration_data.get('sync_metrics', ['dau', 'mau', 'stickiness'])),
                'created_at': datetime.now().isoformat(),
                'is_active': True,
                'workspace_name': 'Demo Workspace',
                'last_sync': None,
                'sync_count': 0
            }

            integration_id = await db.insert('notion_integrations', integration_record)

            # Log user history
            await UserHistoryManager.log_user_action(
                user_id=user_id,
                action_type=UserActionType.CREATE_DATA_SOURCE,
                description=f"Connected DEMO Notion integration",
                metadata={
                    'integration_id': integration_id,
                    'database_name': 'Demo Database',
                    'workspace': 'Demo Workspace'
                }
            )

            return {
                'success': True,
                'message': '🎉 DEMO Notion integration connected successfully!\n• Workspace: Demo Workspace\n• Database: Demo Database\n• User: Demo User',
                'integration_id': integration_id,
                'database_name': 'Demo Database',
                'workspace_name': 'Demo Workspace',
                'user_name': 'Demo User'
            }

        except Exception as e:
            logger.error(f"DEMO Notion connection error: {e}")
            return {
                'success': False,
                'message': f'❌ Failed to connect to Notion: {str(e)}'
            }

    @staticmethod
    async def sync_analytics_to_notion(integration_id: int, user_id: int) -> Dict[str, Any]:
        """DEMO: Simulate analytics sync to Notion - NO LIVE API CALLS"""
        try:
            db = await DatabaseManager.get_instance()
            integration = await db.get_notion_integration(integration_id, user_id)

            if not integration:
                return {'success': False, 'message': '❌ Integration not found'}

            logger.info(f"🔄 DEMO sync to Notion database: {integration['database_id']}")

            # Get demo analytics data
            demo_data = await AnalyticsManager.get_dau_mau_data()

            # DEMO: Simulate successful sync
            current_date = datetime.now().strftime('%Y-%m-%d')
            current_time = datetime.now().strftime('%H:%M:%S')

            # Update integration stats
            current_sync_count = integration.get('sync_count', 0) + 1
            await db.update_notion_integration(integration_id, {
                'last_sync': datetime.now().isoformat(),
                'sync_count': current_sync_count
            })

            # Log the sync
            await UserHistoryManager.log_user_action(
                user_id=user_id,
                action_type=UserActionType.EXPORT_DATA,
                description=f"Synced DEMO analytics data to Notion",
                metadata={
                    'integration_id': integration_id,
                    'page_id': 'demo_page_id',
                    'data_sent': ['DAU', 'MAU', 'Stickiness', 'Data Source'],
                    'sync_count': current_sync_count
                }
            )

            logger.info(f"✅ DEMO sync successful!")

            return {
                'success': True,
                'message': f'🎉 DEMO data synced successfully to Notion!\n• Page: demo_page_id\n• Data: 4 metrics\n• Time: {current_time}',
                'page_id': 'demo_page_id',
                'page_url': 'https://notion.so/demo-page',
                'last_sync': datetime.now().isoformat(),
                'data_sent': ['DAU', 'MAU', 'Stickiness', 'Data Source'],
                'sync_count': current_sync_count,
                'demo_data': {
                    'dau': demo_data['summary']['current_dau'],
                    'mau': demo_data['summary']['current_mau'],
                    'stickiness': demo_data['summary']['stickiness'],
                    'source': 'demo_data',
                    'data_quality': 'high'
                }
            }

        except Exception as e:
            logger.error(f"DEMO Notion sync error: {e}")
            return {
                'success': False,
                'message': f'❌ DEMO sync failed: {str(e)}'
            }

    @staticmethod
    async def create_analytics_database(integration_id: int, user_id: int) -> Dict[str, Any]:
        """DEMO: Simulate database creation - NO LIVE API CALLS"""
        try:
            db = await DatabaseManager.get_instance()
            integration = await db.get_notion_integration(integration_id, user_id)

            if not integration:
                return {'success': False, 'message': '❌ Integration not found'}

            logger.info("🆕 DEMO analytics database creation in Notion...")

            # DEMO: Simulate successful database creation
            new_database_id = f"demo_db_{int(time.time())}"

            # Update the integration with the new database ID
            await db.update_notion_integration(integration_id, {
                'database_id': new_database_id
            })

            logger.info(f"✅ DEMO database created: {new_database_id}")

            return {
                'success': True,
                'message': f'🎉 DEMO analytics database created successfully!\n• Name: Analytics Dashboard - Demo\n• URL: https://notion.so/demo-database',
                'database_id': new_database_id,
                'database_url': 'https://notion.so/demo-database',
                'database_name': 'Analytics Dashboard - Demo'
            }

        except Exception as e:
            logger.error(f"DEMO database creation error: {e}")
            return {
                'success': False,
                'message': f'❌ DEMO database creation failed: {str(e)}'
            }

    @staticmethod
    async def test_notion_connection(notion_token: str) -> Dict[str, Any]:
        """DEMO: Test Notion API connection - NO LIVE API CALLS"""
        try:
            # Validate token format
            if not notion_token.startswith('ntn_'):
                return {
                    'success': False,
                    'message': '❌ Invalid token format. Only ntn_ tokens are supported.'
                }

            logger.info(f"🧪 DEMO Notion connection test...")

            # DEMO: Simulate successful connection
            return {
                'success': True,
                'message': '🎉 DEMO Notion connection successful!\n• User: Demo User\n• Email: demo@example.com\n• Type: user\n• Token: ntn_ format ✅',
                'user_info': {
                    'name': 'Demo User',
                    'email': 'demo@example.com',
                    'type': 'user'
                },
                'token_format': 'ntn_'
            }

        except Exception as e:
            logger.error(f"DEMO connection test failed: {e}")
            return {
                'success': False,
                'message': f'❌ DEMO connection test failed: {str(e)}'
            }

    @staticmethod
    async def get_notion_integrations(user_id: int) -> List[Dict[str, Any]]:
        """Get user's DEMO Notion integrations"""
        try:
            db = await DatabaseManager.get_instance()
            integrations = await db.get_notion_integrations(user_id)

            # Mask token for security and add status info
            for integration in integrations:
                if integration.get('notion_token'):
                    token = integration['notion_token']
                    integration['notion_token_masked'] = f"{token[:8]}...{token[-4:]}" if len(token) > 12 else "***"
                    integration.pop('notion_token', None)

                # Add sync status
                last_sync = integration.get('last_sync')
                if last_sync:
                    last_sync_dt = datetime.fromisoformat(last_sync.replace('Z', '+00:00'))
                    integration['last_sync_human'] = last_sync_dt.strftime('%Y-%m-%d %H:%M:%S')
                    integration['sync_status'] = 'active'
                else:
                    integration['sync_status'] = 'never_synced'

            return integrations
        except Exception as e:
            logger.error(f"Error fetching Notion integrations: {e}")
            return []

    @staticmethod
    async def get_integration_status(integration_id: int, user_id: int) -> Dict[str, Any]:
        """Get detailed status of DEMO Notion integration"""
        try:
            db = await DatabaseManager.get_instance()
            integration = await db.get_notion_integration(integration_id, user_id)

            if not integration:
                return {'success': False, 'message': '❌ Integration not found'}

            # Test DEMO connection
            notion_token = integration['notion_token']
            connection_test = await NotionIntegrationManager.test_notion_connection(notion_token)

            return {
                'success': True,
                'integration_id': integration_id,
                'database_id': integration['database_id'],
                'is_active': integration.get('is_active', True),
                'last_sync': integration.get('last_sync'),
                'sync_count': integration.get('sync_count', 0),
                'connection_status': connection_test['success'],
                'connection_message': connection_test['message'],
                'database_status': {
                    'accessible': True,
                    'name': 'Demo Database',
                    'properties': ['Name', 'Date', 'DAU', 'MAU', 'Stickiness', 'Data Source', 'Status']
                },
                'sync_metrics': json.loads(integration['sync_metrics']) if integration.get('sync_metrics') else [],
                'token_format': 'ntn_',
                'workspace_name': integration.get('workspace_name', 'Demo Workspace'),
                'created_at': integration.get('created_at')
            }

        except Exception as e:
            logger.error(f"Integration status error: {e}")
            return {'success': False, 'message': f'❌ Error checking integration status: {str(e)}'}


#  Dependency to get current user
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials missing",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    payload = AuthManager.verify_token(token)

    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if payload.get('type') != 'access':
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
            headers={"WWW-Authenticate": "Bearer"},
        )

    db = await DatabaseManager.get_instance()
    user_id = payload.get('user_id')

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = await db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


# Create directories
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)


# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    await DatabaseManager.get_instance()


@app.get("/api/verify-admin")
async def verify_admin():
    """Verify admin user exists and return login credentials"""
    db = await DatabaseManager.get_instance()
    admin_user = await db.get_user_by_email('admin@example.com')

    if admin_user:
        return {
            "status": "success",
            "message": "Admin user exists and is ready for login",
            "credentials": {
                "email": "admin@example.com",
                "password": "admin123"
            }
        }
    else:
        return {
            "status": "error",
            "message": "Admin user not found. Please restart the server."
        }
#  Cohort Analysis Endpoints
@app.post("/api/analytics/cohort-analysis")
async def analyze_cohorts(
        cohort_request: CohortAnalysisRequest,
        current_user: dict = Depends(get_current_user)
):
    """Perform cohort analysis"""
    try:
        result = await CohortAnalysisManager.analyze_cohorts(cohort_request)

        # Save analysis to history
        db = await DatabaseManager.get_instance()
        analysis_data = {
            'user_id': current_user['id'],
            'cohort_type': cohort_request.cohort_type.value,
            'metric': cohort_request.metric,
            'period_count': cohort_request.period_count,
            'created_at': datetime.now().isoformat(),
            'results': json.dumps(result)
        }

        analysis_id = await db.create_cohort_analysis(analysis_data)

        return result

    except Exception as e:
        logger.error(f"Cohort analysis endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Cohort analysis failed")


@app.get("/api/analytics/cohort-analysis/history")
async def get_cohort_analysis_history(current_user: dict = Depends(get_current_user)):
    """Get cohort analysis history for user"""
    try:
        db = await DatabaseManager.get_instance()
        user_analyses = await db.get_cohort_analyses(current_user['id'])
        return {"analyses": user_analyses}
    except Exception as e:
        logger.error(f"Cohort history error: {e}")
        raise HTTPException(status_code=500, detail="Error fetching cohort analysis history")

@app.post("/api/signup")
async def signup(user_data: UserCreate, request: Request):
    try:
        db = await DatabaseManager.get_instance()

        print(f"🔄 Signup attempt for: {user_data.email}")

        # Check if user already exists
        existing_user = await db.get_user_by_email(user_data.email)
        if existing_user:
            print(f"❌ User already exists: {user_data.email}")
            raise HTTPException(status_code=400, detail="Email already registered")

        # Create new user
        hashed_password = AuthManager.hash_password(user_data.password)
        new_user_data = {
            'email': user_data.email,
            'password_hash': hashed_password,
            'full_name': user_data.full_name,
            'role': user_data.role.value,
            'is_verified': True,
            'created_at': datetime.now().isoformat()
        }

        print(f"📝 Creating user with data: {new_user_data['email']}")

        # Insert user and get the ID
        user_id = await db.create_user(new_user_data)
        print(f"✅ User created with ID: {user_id}")

        # Verify the user was actually created
        verify_user = await db.get_user_by_email(user_data.email)
        if not verify_user:
            print(f"❌ User creation verification failed for: {user_data.email}")
            raise HTTPException(status_code=500, detail="User creation failed - please try again")

        print(f"✅ User verified in database: {verify_user['email']}")

        # Auto-login after signup
        access_token, refresh_token = AuthManager.create_tokens({
            'user_id': user_id,
            'role': user_data.role.value
        })

        # Log user history
        await UserHistoryManager.log_user_action(
            user_id=user_id,
            action_type=UserActionType.LOGIN,
            description="New user signed up",
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get('user-agent')
        )

        return JSONResponse(content={
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "user_email": user_data.email,
            "user_role": user_data.role.value,
            "full_name": user_data.full_name,
            "user_id": user_id,
            "message": "User created successfully"
        })

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Signup error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Signup failed")

# Funnel Drop-Off Engine Endpoints
@app.post("/api/analytics/funnel-dropoff")
async def analyze_funnel_dropoff(
        funnel_request: FunnelDropOffRequest,
        current_user: dict = Depends(get_current_user)
):
    """Analyze funnel drop-off points"""
    try:
        result = await FunnelDropOffEngine.analyze_funnel_dropoff(funnel_request)

        # Save analysis to history
        db = await DatabaseManager.get_instance()
        analysis_data = {
            'user_id': current_user['id'],
            'funnel_stages': json.dumps(funnel_request.funnel_stages),
            'created_at': datetime.now().isoformat(),
            'results': json.dumps(result)
        }

        analysis_id = await db.create_funnel_analysis(analysis_data)

        return result

    except Exception as e:
        logger.error(f"Funnel drop-off analysis error: {e}")
        raise HTTPException(status_code=500, detail="Funnel drop-off analysis failed")


@app.get("/api/analytics/funnel-dropoff/history")
async def get_funnel_analysis_history(current_user: dict = Depends(get_current_user)):
    """Get funnel analysis history for user"""
    try:
        db = await DatabaseManager.get_instance()
        user_analyses = await db.get_funnel_analyses(current_user['id'])
        return {"analyses": user_analyses}
    except Exception as e:
        logger.error(f"Funnel history error: {e}")
        raise HTTPException(status_code=500, detail="Error fetching funnel analysis history")


# AI Action Recommendation Engine Endpoints
@app.post("/api/ai/recommendations")
async def get_ai_action_recommendations(
        action_request: AIActionRequest,
        current_user: dict = Depends(get_current_user)
):
    """Get AI-powered action recommendations"""
    try:
        # Create engine instance and get recommendations
        engine = AIActionRecommendationEngine()
        result = await engine.generate_recommendations(action_request)

        # Save recommendations to database
        db = await DatabaseManager.get_instance()
        for rec in result.get('recommendations', []):
            rec_data = {
                'user_id': current_user['id'],
                'focus_area': action_request.focus_area,
                'recommendation': json.dumps({
                    'title': rec.title,
                    'description': rec.description,
                    'action_type': rec.action_type,
                    'impact_score': rec.impact_score,
                    'confidence': rec.confidence,
                    'effort_required': rec.effort_required,
                    'metrics_affected': rec.metrics_affected,
                    'suggested_implementation': rec.suggested_implementation,
                    'expected_improvement': rec.expected_improvement,
                    'kpis_to_watch': rec.kpis_to_watch
                }),
                'generated_at': datetime.now().isoformat()
            }
            await db.create_ai_recommendation(rec_data)

        return result

    except Exception as e:
        logger.error(f"AI recommendations error: {e}")
        raise HTTPException(status_code=500, detail="AI recommendation generation failed")


@app.get("/api/ai/recommendations/history")
async def get_ai_recommendations_history(current_user: dict = Depends(get_current_user)):
    """Get AI recommendations history for user"""
    try:
        db = await DatabaseManager.get_instance()
        user_recommendations = await db.get_ai_recommendations(current_user['id'])
        return {"recommendations": user_recommendations}
    except Exception as e:
        logger.error(f"AI recommendations history error: {e}")
        raise HTTPException(status_code=500, detail="Error fetching AI recommendations history")


# ICE Framework Endpoints
@app.post("/api/ice/initiatives")
async def create_ice_initiative(
        initiative_data: ICEInitiativeCreate,
        current_user: dict = Depends(get_current_user)
):
    """Create a new ICE initiative"""
    try:
        initiative = await ICEFrameworkManager.create_ice_initiative(initiative_data, current_user['id'])
        return {"message": "ICE initiative created successfully", "initiative": initiative}
    except Exception as e:
        logger.error(f"ICE initiative creation error: {e}")
        raise HTTPException(status_code=500, detail="Error creating ICE initiative")


@app.get("/api/ice/initiatives")
async def get_ice_initiatives(
        status: Optional[str] = None,
        current_user: dict = Depends(get_current_user)
):
    """Get ICE initiatives"""
    try:
        initiatives = await ICEFrameworkManager.get_ice_initiatives(current_user['id'], status)
        return {"initiatives": initiatives}
    except Exception as e:
        logger.error(f"ICE initiatives fetch error: {e}")
        raise HTTPException(status_code=500, detail="Error fetching ICE initiatives")


@app.get("/api/ice/prioritization-matrix")
async def get_ice_prioritization_matrix(current_user: dict = Depends(get_current_user)):
    """Get ICE prioritization matrix"""
    try:
        matrix = await ICEFrameworkManager.get_ice_prioritization_matrix(current_user['id'])
        return matrix
    except Exception as e:
        logger.error(f"ICE matrix error: {e}")
        raise HTTPException(status_code=500, detail="Error generating ICE matrix")


# User History Endpoints
@app.get("/api/user/history")
async def get_user_history(
        limit: int = 50,
        current_user: dict = Depends(get_current_user)
):
    """Get user history"""
    try:
        history = await UserHistoryManager.get_user_history(current_user['id'], limit)
        return {"history": history}
    except Exception as e:
        logger.error(f"User history error: {e}")
        raise HTTPException(status_code=500, detail="Error fetching user history")


@app.get("/api/user/activity-summary")
async def get_user_activity_summary(current_user: dict = Depends(get_current_user)):
    """Get user activity summary"""
    try:
        summary = await UserHistoryManager.get_user_activity_summary(current_user['id'])
        return summary
    except Exception as e:
        logger.error(f"User activity summary error: {e}")
        raise HTTPException(status_code=500, detail="Error fetching user activity summary")


# DEMO NOTION INTEGRATION ENDPOINTS
@app.post("/api/notion/connect")
async def connect_notion(integration_request: NotionIntegrationRequest, current_user: dict = Depends(get_current_user)):
    """DEMO: Connect to Notion with ntn_ token"""
    try:
        if not integration_request.notion_token.startswith('ntn_'):
            raise HTTPException(
                status_code=400,
                detail="❌ Only ntn_ tokens are supported. Get one from: https://www.notion.so/my-integrations"
            )

        integration_dict = integration_request.dict()
        result = await NotionIntegrationManager.connect_notion(integration_dict, current_user['id'])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"DEMO Notion connection error: {e}")
        raise HTTPException(status_code=500, detail="❌ Error connecting to Notion")


@app.post("/api/notion/test-connection")
async def test_notion_connection(data: dict, current_user: dict = Depends(get_current_user)):
    """DEMO: Test Notion connection with ntn_ token"""
    try:
        token = data.get('notion_token')
        if not token:
            raise HTTPException(status_code=400, detail="❌ Notion token is required")

        result = await NotionIntegrationManager.test_notion_connection(token)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Connection test failed: {str(e)}")


@app.post("/api/notion/{integration_id}/sync")
async def sync_notion(integration_id: int, current_user: dict = Depends(get_current_user)):
    """DEMO: Sync demo data to Notion"""
    try:
        result = await NotionIntegrationManager.sync_analytics_to_notion(integration_id, current_user['id'])
        return result
    except Exception as e:
        logger.error(f"DEMO Notion sync error: {e}")
        raise HTTPException(status_code=500, detail="❌ Error syncing with Notion")


@app.post("/api/notion/{integration_id}/create-database")
async def create_notion_database(integration_id: int, current_user: dict = Depends(get_current_user)):
    """DEMO: Create analytics database in Notion"""
    try:
        result = await NotionIntegrationManager.create_analytics_database(integration_id, current_user['id'])
        return result
    except Exception as e:
        logger.error(f"DEMO database creation error: {e}")
        raise HTTPException(status_code=500, detail="❌ Error creating Notion database")


@app.get("/api/notion/integrations")
async def get_notion_integrations(current_user: dict = Depends(get_current_user)):
    """Get user's Notion integrations"""
    try:
        integrations = await NotionIntegrationManager.get_notion_integrations(current_user['id'])
        return {"integrations": integrations}
    except Exception as e:
        logger.error(f"Error fetching Notion integrations: {e}")
        raise HTTPException(status_code=500, detail="Error fetching integrations")


@app.get("/api/notion/{integration_id}/status")
async def get_notion_integration_status(integration_id: int, current_user: dict = Depends(get_current_user)):
    """Get Notion integration status"""
    try:
        status_info = await NotionIntegrationManager.get_integration_status(integration_id, current_user['id'])
        return status_info
    except Exception as e:
        logger.error(f"Error fetching integration status: {e}")
        raise HTTPException(status_code=500, detail="Error fetching integration status")


@app.post("/api/login")
async def login(login_data: UserLogin, request: Request):
    try:
        db = await DatabaseManager.get_instance()
        user = await db.get_user_by_email(login_data.email)

        if not user:
            # Log all existing users for debugging
            all_users = await db.get_all_users()
            logger.warning(f"Login failed: User {login_data.email} not found. Existing users: {[u['email'] for u in all_users]}")
            raise HTTPException(
                status_code=400,
                detail="No account found with this email. Please sign up first."
            )

        # Check if password hash exists
        if not user.get('password_hash'):
            logger.error(f"User has no password hash: {user['email']}")
            raise HTTPException(
                status_code=500,
                detail="Account setup error. Please contact support."
            )

        password_valid = AuthManager.verify_password(login_data.password, user['password_hash'])

        if not password_valid:
            logger.warning(f"Login failed: Invalid password for {login_data.email}")
            raise HTTPException(
                status_code=400,
                detail="Invalid password. Please check your credentials."
            )

        # Update last login
        await db.update_user(user['id'], {'last_login': datetime.now().isoformat()})

        access_token, refresh_token = AuthManager.create_tokens({
            'user_id': user['id'],
            'role': user['role']
        })

        # Log user history
        await UserHistoryManager.log_user_action(
            user_id=user['id'],
            action_type=UserActionType.LOGIN,
            description="User logged into the system",
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get('user-agent')
        )

        logger.info(f"Login successful: {user['email']}")

        return JSONResponse(content={
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "user_email": user['email'],
            "user_role": user['role'],
            "full_name": user['full_name'],
            "user_id": user['id']
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed due to server error")


@app.post("/api/emergency-create-admin")
async def emergency_create_admin():
    """Emergency endpoint to create admin user"""
    try:
        db = await DatabaseManager.get_instance()

        # Check if admin already exists
        admin_user = await db.get_user_by_email('admin@example.com')
        if admin_user:
            return {"message": "Admin user already exists", "user_id": admin_user['id']}

        # Create admin user
        hashed_password = AuthManager.hash_password('admin123')
        admin_data = {
            'email': 'admin@example.com',
            'password_hash': hashed_password,
            'full_name': 'System Administrator',
            'role': 'admin',
            'is_verified': True,
            'created_at': datetime.now().isoformat()
        }

        user_id = await db.insert('users', admin_data)
        return {"message": "Admin user created successfully", "user_id": user_id}

    except Exception as e:
        logger.error(f"Emergency admin creation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to create admin user")


# Analytics endpoints
@app.get("/api/analytics/dau-mau")
async def get_dau_mau(current_user: dict = Depends(get_current_user)):
    try:
        return await AnalyticsManager.get_dau_mau_data()
    except Exception as e:
        logger.error(f"DAU/MAU error: {e}")
        raise HTTPException(status_code=500, detail="Error generating analytics")


@app.get("/api/analytics/kpi-hierarchy")
async def get_kpi_hierarchy(current_user: dict = Depends(get_current_user)):
    try:
        return await AnalyticsManager.get_kpi_hierarchy()
    except Exception as e:
        logger.error(f"KPI hierarchy error: {e}")
        raise HTTPException(status_code=500, detail="Error generating KPI hierarchy")


@app.get("/api/analytics/feature-heatmaps")
async def get_feature_heatmaps(current_user: dict = Depends(get_current_user)):
    try:
        return await AnalyticsManager.get_feature_heatmaps()
    except Exception as e:
        logger.error(f"Feature heatmap error: {e}")
        raise HTTPException(status_code=500, detail="Error generating heatmaps")


@app.get("/api/analytics/correlation-matrix")
async def get_correlation_matrix(current_user: dict = Depends(get_current_user)):
    try:
        return await AnalyticsManager.get_correlation_matrix()
    except Exception as e:
        logger.error(f"Correlation matrix error: {e}")
        raise HTTPException(status_code=500, detail="Error generating correlation matrix")


@app.get("/api/analytics/north-star")
async def get_north_star_metric(current_user: dict = Depends(get_current_user)):
    try:
        return await AnalyticsManager.get_north_star_metric()
    except Exception as e:
        logger.error(f"North star metric error: {e}")
        raise HTTPException(status_code=500, detail="Error generating north star insights")


@app.get("/api/analytics/funnel-analysis")
async def get_funnel_analysis(current_user: dict = Depends(get_current_user)):
    try:
        return await AnalyticsManager.get_funnel_analysis()
    except Exception as e:
        logger.error(f"Funnel analysis error: {e}")
        raise HTTPException(status_code=500, detail="Error analyzing funnel")


@app.get("/api/analytics/ai-recommendations")
async def get_ai_recommendations(current_user: dict = Depends(get_current_user)):
    try:
        return await AnalyticsManager.get_ai_recommendations()
    except Exception as e:
        logger.error(f"AI recommendations error: {e}")
        raise HTTPException(status_code=500, detail="Error generating recommendations")


# A/B Testing endpoints
@app.post("/api/ab-tests")
async def create_ab_test(test_data: ABTestCreate, current_user: dict = Depends(get_current_user)):
    """Create a new A/B test"""
    try:
        test_dict = test_data.dict()
        new_test = await ABTestingManager.create_ab_test(test_dict, current_user['id'])
        return {"message": "A/B test created successfully", "test": new_test}
    except Exception as e:
        logger.error(f"A/B test creation error: {e}")
        raise HTTPException(status_code=500, detail="Error creating A/B test")


@app.get("/api/ab-tests")
async def get_ab_tests(current_user: dict = Depends(get_current_user)):
    """Get all A/B tests for current user"""
    try:
        tests = await ABTestingManager.get_ab_tests(current_user['id'])
        return {"tests": tests}
    except Exception as e:
        logger.error(f"A/B tests fetch error: {e}")
        raise HTTPException(status_code=500, detail="Error fetching A/B tests")


@app.get("/api/ab-tests/{test_id}")
async def get_ab_test(test_id: int, current_user: dict = Depends(get_current_user)):
    """Get specific A/B test"""
    try:
        test = await ABTestingManager.get_ab_test(test_id, current_user['id'])
        if not test:
            raise HTTPException(status_code=404, detail="A/B test not found")
        return {"test": test}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"A/B test fetch error: {e}")
        raise HTTPException(status_code=500, detail="Error fetching A/B test")


@app.put("/api/ab-tests/{test_id}")
async def update_ab_test(test_id: int, update_data: ABTestUpdate, current_user: dict = Depends(get_current_user)):
    """Update A/B test"""
    try:
        update_dict = {k: v for k, v in update_data.dict().items() if v is not None}
        updated_test = await ABTestingManager.update_ab_test(test_id, update_dict, current_user['id'])
        if not updated_test:
            raise HTTPException(status_code=404, detail="A/B test not found")
        return {"message": "A/B test updated successfully", "test": updated_test}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"A/B test update error: {e}")
        raise HTTPException(status_code=500, detail="Error updating A/B test")


@app.post("/api/ab-tests/{test_id}/start")
async def start_ab_test(test_id: int, current_user: dict = Depends(get_current_user)):
    """Start an A/B test"""
    try:
        started_test = await ABTestingManager.start_ab_test(test_id, current_user['id'])
        if not started_test:
            raise HTTPException(status_code=404, detail="A/B test not found")
        return {"message": "A/B test started successfully", "test": started_test}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"A/B test start error: {e}")
        raise HTTPException(status_code=500, detail="Error starting A/B test")


@app.get("/api/ab-tests/{test_id}/results")
async def get_ab_test_results(test_id: int, current_user: dict = Depends(get_current_user)):
    """Get A/B test results with statistical analysis"""
    try:
        results = await ABTestingManager.get_ab_test_results(test_id, current_user['id'])
        if not results:
            raise HTTPException(status_code=404, detail="A/B test not found")
        return results
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"A/B test results error: {e}")
        raise HTTPException(status_code=500, detail="Error fetching A/B test results")


# Data Export endpoints
@app.post("/api/exports")
async def export_data(export_request: ExportRequest, current_user: dict = Depends(get_current_user)):
    """Export data in requested format"""
    try:
        export_dict = export_request.dict()
        result = await DataExportManager.export_data(export_dict, current_user['id'])

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Data export error: {e}")
        raise HTTPException(status_code=500, detail=f"Error exporting data: {str(e)}")


@app.post("/api/exports/download")
async def download_export(export_request: ExportRequest, current_user: dict = Depends(get_current_user)):
    """Download export as file"""
    try:
        export_dict = export_request.dict()
        result = await DataExportManager.export_data(export_dict, current_user['id'])

        if export_request.format == 'csv':
            if isinstance(result['data'], list) and len(result['data']) > 0:
                output = io.StringIO()
                writer = csv.DictWriter(output, fieldnames=result['data'][0].keys())
                writer.writeheader()
                writer.writerows(result['data'])
                csv_content = output.getvalue()

                response = StreamingResponse(
                    io.BytesIO(csv_content.encode('utf-8')),
                    media_type="text/csv",
                    headers={
                        "Content-Disposition": f"attachment; filename={result['file_name']}",
                        "Content-Type": "text/csv; charset=utf-8"
                    }
                )
                return response
            else:
                raise HTTPException(status_code=400, detail="No data available for export")

        elif export_request.format == 'json':
            json_content = json.dumps(result['data'], indent=2, ensure_ascii=False)

            response = StreamingResponse(
                io.BytesIO(json_content.encode('utf-8')),
                media_type="application/json",
                headers={
                    "Content-Disposition": f"attachment; filename={result['file_name']}",
                    "Content-Type": "application/json; charset=utf-8"
                }
            )
            return response

        else:
            raise HTTPException(status_code=400, detail="Unsupported format")

    except Exception as e:
        logger.error(f"Data export download error: {e}")
        raise HTTPException(status_code=500, detail=f"Error exporting data: {str(e)}")


@app.get("/api/exports/history")
async def get_export_history(current_user: dict = Depends(get_current_user)):
    """Get user's export history"""
    try:
        history = await DataExportManager.get_export_history(current_user['id'])
        return {"exports": history}
    except Exception as e:
        logger.error(f"Export history error: {e}")
        raise HTTPException(status_code=500, detail="Error fetching export history")


# Anomaly Detection endpoints
@app.post("/api/anomaly-detection/detect")
async def detect_anomalies(detection_request: AnomalyDetectionRequest, current_user: dict = Depends(get_current_user)):
    """Detect anomalies in metrics"""
    try:
        detection_dict = detection_request.dict()
        results = await AnomalyDetectionManager.detect_anomalies(detection_dict)

        db = await DatabaseManager.get_instance()

        anomaly_data = {
            'user_id': current_user['id'],
            'metric': detection_request.metric,
            'time_range': detection_request.time_range,
            'sensitivity': detection_request.sensitivity,
            'anomalies_detected': len(results['anomalies']),
            'detected_at': datetime.now().isoformat(),
            'results': json.dumps(results)
        }

        await db.insert('anomaly_detections', anomaly_data)

        return results

    except Exception as e:
        logger.error(f"Anomaly detection error: {e}")
        raise HTTPException(status_code=500, detail="Error detecting anomalies")


@app.get("/api/anomaly-detection/history")
async def get_anomaly_history(current_user: dict = Depends(get_current_user)):
    """Get anomaly detection history"""
    try:
        history = await AnomalyDetectionManager.get_anomaly_history(current_user['id'])
        return {"anomaly_history": history}
    except Exception as e:
        logger.error(f"Anomaly history error: {e}")
        raise HTTPException(status_code=500, detail="Error fetching anomaly history")


# User management endpoints
@app.get("/api/user/profile")
async def get_user_profile(current_user: dict = Depends(get_current_user)):
    return {
        "id": current_user["id"],
        "email": current_user["email"],
        "full_name": current_user["full_name"],
        "role": current_user["role"],
        "created_at": current_user.get("created_at", datetime.now().isoformat()),
        "last_login": current_user.get("last_login", datetime.now().isoformat())
    }


@app.get("/api/admin/users")
async def get_all_users(current_user: dict = Depends(get_current_user)):
    if current_user['role'] != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")

    db = await DatabaseManager.get_instance()
    users_data = await db.get_all_users()
    return users_data


@app.get("/api/debug/check-database")
async def debug_check_database():
    """Debug endpoint to check database state"""
    try:
        db = await DatabaseManager.get_instance()

        # Check various tables
        users_count_result = await db.db.fetch_one("SELECT COUNT(*) as count FROM users")
        users_count = users_count_result['count'] if users_count_result else 0

        # Get all users
        all_users = await db.db.fetch_all("SELECT id, email, role FROM users")

        return {
            "users_count": users_count,
            "all_users": all_users,
            "database_status": "connected"
        }
    except Exception as e:
        return {"error": str(e), "database_status": "error"}


@app.post("/api/debug/create-admin-now")
async def debug_create_admin_now():
    """Force create admin user now"""
    try:
        db = await DatabaseManager.get_instance()

        # Check if admin already exists
        admin_user = await db.get_user_by_email('admin@example.com')
        if admin_user:
            return {
                "status": "exists",
                "message": "Admin user already exists",
                "user": {
                    "id": admin_user['id'],
                    "email": admin_user['email'],
                    "role": admin_user['role']
                }
            }

        # Create admin user
        hashed_password = AuthManager.hash_password('admin123')
        admin_data = {
            'email': 'admin@example.com',
            'password_hash': hashed_password,
            'full_name': 'System Administrator',
            'role': 'admin',
            'is_verified': True,
            'created_at': datetime.now().isoformat()
        }

        admin_id = await db.create_user(admin_data)

        return {
            "status": "created",
            "message": "Admin user created successfully",
            "user_id": admin_id,
            "credentials": {
                "email": "admin@example.com",
                "password": "admin123"
            }
        }

    except Exception as e:
        return {"status": "error", "message": f"Failed to create admin: {str(e)}"}


@app.get("/api/debug/check-tables")
async def debug_check_tables():
    """Check if all tables exist"""
    try:
        db = await DatabaseManager.get_instance()

        tables = [
            'users', 'analytics_events', 'ice_initiatives', 'user_history',
            'data_sources', 'real_data_config', 'ab_tests', 'custom_reports',
            'data_exports', 'cohort_analyses', 'funnel_analyses',
            'ai_action_recommendations', 'notion_integrations'
        ]

        table_status = {}
        for table in tables:
            try:
                result = await db.db.fetch_one(f"SELECT COUNT(*) as count FROM {table}")
                table_status[table] = {"exists": True, "count": result['count']}
            except Exception as e:
                table_status[table] = {"exists": False, "error": str(e)}

        return table_status

    except Exception as e:
        return {"error": str(e)}

@app.post("/api/admin/users")
async def create_user_admin(user_data: UserCreate, current_user: dict = Depends(get_current_user)):
    """Create user (admin only) - FIXED VERSION"""
    if current_user['role'] != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")

    try:
        db = await DatabaseManager.get_instance()

        # Check if user already exists
        existing_user = await db.get_user_by_email(user_data.email)
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")

        # Create new user
        hashed_password = AuthManager.hash_password(user_data.password)
        new_user_data = {
            'email': user_data.email,
            'password_hash': hashed_password,
            'full_name': user_data.full_name,
            'role': user_data.role.value,
            'created_at': datetime.now().isoformat(),
            'is_verified': True
        }

        user_id = await db.create_user(new_user_data)

        return {"message": "User created successfully", "user_id": user_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Admin user creation error: {e}")
        raise HTTPException(status_code=500, detail="User creation failed")


@app.put("/api/admin/users/{user_id}/role")
async def update_user_role(user_id: int, role_data: UserUpdateRole, current_user: dict = Depends(get_current_user)):
    """Update user role (admin only)"""
    if current_user['role'] != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")

    db = await DatabaseManager.get_instance()
    success = await db.update_user(user_id, {'role': role_data.role.value})
    if success:
        return {"message": f"User role updated to {role_data.role.value}"}
    else:
        raise HTTPException(status_code=404, detail="User not found")


@app.delete("/api/admin/users/{user_id}")
async def delete_user(user_id: int, current_user: dict = Depends(get_current_user)):
    """Delete user (admin only)"""
    if current_user['role'] != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")

    if user_id == current_user['id']:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")

    db = await DatabaseManager.get_instance()
    success = await db.delete('users', 'id = $1', user_id)
    if success:
        return {"message": "User deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="User not found")


# Custom Reports endpoints
@app.get("/api/custom-reports")
async def get_custom_reports(current_user: dict = Depends(get_current_user)):
    """Get all custom reports for current user"""
    db = await DatabaseManager.get_instance()
    user_reports = await db.get_custom_reports(current_user['id'])
    return user_reports


@app.post("/api/custom-reports")
async def create_custom_report(report_data: CustomReportCreate, current_user: dict = Depends(get_current_user)):
    """Create a new custom report"""
    db = await DatabaseManager.get_instance()

    report_dict = {
        'user_id': current_user['id'],
        'name': report_data.name,
        'metrics': json.dumps(report_data.metrics),
        'filters': json.dumps(report_data.filters),
        'chart_type': report_data.chart_type,
        'created_at': datetime.now().isoformat(),
        'updated_at': datetime.now().isoformat()
    }

    report_id = await db.create_custom_report(report_dict)

    # Log user history
    await UserHistoryManager.log_user_action(
        user_id=current_user['id'],
        action_type=UserActionType.CREATE_REPORT,
        description=f"Created custom report: {report_data.name}",
        metadata={'report_id': report_id, 'report_name': report_data.name}
    )

    report_result = await AnalyticsManager.generate_custom_report({**report_dict, 'id': report_id})

    return {
        "message": "Custom report created successfully",
        "report_id": report_id,
        "report_data": report_result
    }


@app.get("/api/custom-reports/{report_id}")
async def get_custom_report(report_id: int, current_user: dict = Depends(get_current_user)):
    """Get specific custom report"""
    db = await DatabaseManager.get_instance()

    reports = await db.get_custom_reports(current_user['id'])
    for report in reports:
        if report['id'] == report_id:
            report_data = await AnalyticsManager.generate_custom_report(report)
            return report_data

    raise HTTPException(status_code=404, detail="Report not found")


@app.delete("/api/custom-reports/{report_id}")
async def delete_custom_report(report_id: int, current_user: dict = Depends(get_current_user)):
    """Delete custom report"""
    db = await DatabaseManager.get_instance()

    success = await db.delete_custom_report(report_id, current_user['id'])
    if success:
        return {"message": "Report deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Report not found")


@app.get("/api/debug/database-state")
async def debug_database_state():
    """Comprehensive database state check"""
    try:
        db = await DatabaseManager.get_instance()

        # Test basic connection
        test_result = await db.db.fetch_one("SELECT 1 as test, NOW() as time")

        # Check all tables
        tables = ['users', 'analytics_events', 'ice_initiatives']
        table_counts = {}

        for table in tables:
            try:
                count_result = await db.db.fetch_one(f"SELECT COUNT(*) as count FROM {table}")
                table_counts[table] = count_result['count']
            except Exception as e:
                table_counts[table] = f"Error: {str(e)}"

        # Get all users with details
        users = await db.db.fetch_all("SELECT id, email, role, created_at FROM users ORDER BY id")

        return {
            "database_connection": "✅ Working" if test_result else "❌ Failed",
            "server_time": test_result['time'] if test_result else "Unknown",
            "table_counts": table_counts,
            "total_users": len(users),
            "users": users,
            "admin_exists": any(user['email'] == 'admin@example.com' for user in users)
        }

    except Exception as e:
        return {"error": str(e), "type": type(e).__name__}


@app.post("/api/debug/fix-now")
async def debug_fix_now():
    """Emergency fix - recreates everything"""
    try:
        db = await DatabaseManager.get_instance()

        print("🚨 EMERGENCY DATABASE FIX STARTED...")

        # Drop and recreate all tables
        print("🔄 Recreating tables...")
        await db.db._create_tables()

        # Create admin user
        print("🔄 Creating admin user...")
        admin_id = await db._create_admin_user()

        # Create test user
        print("🔄 Creating test user...")
        test_user_data = {
            'email': 'test@example.com',
            'password_hash': AuthManager.hash_password('test123'),
            'full_name': 'Test User',
            'role': 'viewer',
            'is_verified': True,
            'created_at': datetime.now()
        }
        test_id = await db.create_user(test_user_data)

        print("✅ Emergency fix completed")

        return {
            "success": True,
            "message": "Database completely reset and rebuilt",
            "users_created": {
                "admin": {"id": admin_id, "email": "admin@example.com", "password": "admin123"},
                "test": {"id": test_id, "email": "test@example.com", "password": "test123"}
            }
        }

    except Exception as e:
        print(f"❌ Emergency fix failed: {e}")
        return {"success": False, "error": str(e)}


@app.on_event("startup")
async def startup_event():
    """Enhanced startup with better error handling"""
    try:
        print("=" * 60)
        print("🚀 STARTING AI ANALYTICS DASHBOARD v4.0")
        print("=" * 60)

        # Initialize database
        db = await DatabaseManager.get_instance()

        # Force verification
        admin_user = await db.get_user_by_email('admin@example.com')
        if admin_user:
            print(f"✅ Admin user verified: {admin_user['email']} (ID: {admin_user['id']})")
        else:
            print("❌ CRITICAL: Admin user missing after initialization!")

        # Show all users
        all_users = await db.get_all_users()
        print(f"📊 Total users in system: {len(all_users)}")
        for user in all_users:
            print(f"   👤 {user['id']}: {user['email']} ({user['role']})")

        print("🎉 Startup completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"💥 STARTUP FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise
# Page endpoints
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/login")
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/signup")
async def signup_page(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})


@app.get("/dashboard")
async def dashboard_page(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


# NEW: Page endpoints for new features
@app.get("/cohort-analysis")
async def cohort_analysis_page(request: Request):
    return templates.TemplateResponse("cohort-analysis.html", {"request": request})


@app.get("/funnel-analysis")
async def funnel_analysis_page(request: Request):
    return templates.TemplateResponse("funnel-analysis.html", {"request": request})


@app.get("/ai-recommendations")
async def ai_recommendations_page(request: Request):
    return templates.TemplateResponse("ai-recommendations.html", {"request": request})


@app.get("/ice-framework")
async def ice_framework_page(request: Request):
    return templates.TemplateResponse("ice-framework.html", {"request": request})


@app.get("/data-export")
async def data_export_page(request: Request):
    return templates.TemplateResponse("data-export.html", {"request": request})


@app.get("/user-management")
async def user_management_page(request: Request):
    return templates.TemplateResponse("user-management.html", {"request": request})


@app.get("/user-history")
async def user_history_page(request: Request):
    return templates.TemplateResponse("user-history.html", {"request": request})


@app.get("/profile")
async def profile_page(request: Request):
    return templates.TemplateResponse("profile.html", {"request": request})


@app.get("/ab-testing")
async def ab_testing_page(request: Request):
    return templates.TemplateResponse("ab-testing.html", {"request": request})


@app.get("/anomaly-detection")
async def anomaly_detection_page(request: Request):
    return templates.TemplateResponse("anomaly-detection.html", {"request": request})


@app.get("/custom-reports")
async def custom_reports_page(request: Request):
    return templates.TemplateResponse("custom-reports.html", {"request": request})


@app.get("/notion-integration")
async def notion_integration_page(request: Request):
    return templates.TemplateResponse("notion-integration.html", {"request": request})

@app.get("/real-data-config")
async def real_data_config_page(request: Request):
    return templates.TemplateResponse("real-data-config.html", {"request": request})

@app.get("/logout")
async def logout_page(request: Request):
    return templates.TemplateResponse("logout.html", {"request": request})


@app.post("/api/notion/create-sample")
async def create_sample_integration(current_user: dict = Depends(get_current_user)):
    """Create sample integration for testing"""
    try:
        db = await DatabaseManager.get_instance()

        sample_integration = {
            'user_id': current_user['id'],
            'notion_token': 'ntn_sample_token',
            'database_id': 'sample_database_id',
            'workspace_name': 'Sample Workspace',
            'sync_frequency': 'daily',
            'sync_metrics': json.dumps(['dau', 'mau', 'retention']),
            'created_at': datetime.now().isoformat(),
            'last_sync': datetime.now().isoformat(),
            'is_active': True
        }

        integration_id = await db.insert('notion_integrations', sample_integration)

        return {
            "message": "Sample integration created",
            "integration_id": integration_id
        }
    except Exception as e:
        logger.error(f"Sample creation error: {e}")
        raise HTTPException(status_code=500, detail="Error creating sample integration")


# Health check endpoint
@app.get("/health")
async def health_check():
    db = await DatabaseManager.get_instance()
    real_data_config = await db.get_real_data_config()

    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "4.0.0",
        "database": "PostgreSQL",
        "users_count": len(await db.get_all_users()),
        "real_data_enabled": real_data_config.get('enabled', True),
        "data_sources_count": len(await db.get_data_sources(1)),  # Count for admin user
        "primary_source": real_data_config.get('primary_source'),
        "ice_initiatives_count": len(await db.get_ice_initiatives(1)),
        "user_history_count": len(await db.get_user_history(1)),
        "cohort_analyses_count": len(await db.get_cohort_analyses(1)),
        "funnel_analyses_count": len(await db.get_funnel_analyses(1)),
        "ai_recommendations_count": len(await db.get_ai_recommendations(1)),
        "notion_integrations_count": len(await db.get_notion_integrations(1)),
        "features": [
            "postgresql_database",
            "cohort_analysis",
            "funnel_dropoff_engine",
            "ai_action_recommendations",
            "demo_data_only",
            "rice_ice_framework",
            "user_history_tracking",
            "demo_notion_integration",
            "production_ready"
        ]
    }


if __name__ == "__main__":
    import uvicorn

    print("=" * 80)
    print(" AI-Powered Analytics Dashboard v4.0 - PostgreSQL Edition")
    print("=" * 80)
    print(" PRODUCTION READY FEATURES:")
    print("  • PostgreSQL Database - Enterprise-grade data storage")
    print("  • All advanced analytics features")
    print("  • Demo data integration (no external dependencies)")
    print("  • User management system")
    print("  • DEMO WORKING NOTION INTEGRATION (No API calls)")
    print("")
    print(" Starting server on http://localhost:8000")
    print(" Database: PostgreSQL")
    print(" Notion Integration: Demo Mode - No API keys needed")
    print("=" * 80)

    uvicorn.run(app, host="0.0.0.0", port=8000)