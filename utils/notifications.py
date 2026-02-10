"""
System notifications module for Windows.
Sends desktop notifications when unknown faces are detected.
"""

import os
import sys

def send_notification(title, message, duration=5000):
    """
    Send a Windows system notification.
    
    Args:
        title (str): Notification title
        message (str): Notification message
        duration (int): Display duration in milliseconds (default: 5000ms)
    """
    try:
        # Try using win10toast (if installed)
        try:
            from win10toast import ToastNotifier
            toaster = ToastNotifier()
            toaster.show_toast(title, message, duration=duration // 1000, threaded=True)
            return True
        except ImportError:
            pass
        
        # Fallback: Use Windows powershell for notifications
        ps_command = f'''
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
[Windows.UI.Notifications.ToastNotification, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
[Windows.Data.Xml.Dom.XmlDocument, System.Xml.XmlDocument, ContentType = WindowsRuntime] | Out-Null

$APP_ID = 'FacialRecognition'

$template = @"
<toast>
    <visual>
        <binding template="ToastText02">
            <text id="1">{title}</text>
            <text id="2">{message}</text>
        </binding>
    </visual>
</toast>
"@

$xml = New-Object Windows.Data.Xml.Dom.XmlDocument
$xml.LoadXml($template)
$toast = New-Object Windows.UI.Notifications.ToastNotification $xml
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($APP_ID).Show($toast)
'''
        
        # Save and execute PowerShell script
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ps1', delete=False) as f:
            f.write(ps_command)
            temp_ps_file = f.name
        
        try:
            os.system(f'powershell -NoProfile -ExecutionPolicy Bypass -File "{temp_ps_file}"')
            return True
        finally:
            os.unlink(temp_ps_file)
            
    except Exception as e:
        print(f"Notification error: {e}")
        return False


def notify_unknown_face():
    """Send a notification for unknown face detection."""
    title = "⚠️ Security Alert"
    message = "An unknown face tried to access."
    return send_notification(title, message, duration=7000)


def notify_known_face(name):
    """Send a notification for known face detection."""
    title = "✓ Face Recognized"
    message = f"Welcome, {name}!"
    return send_notification(title, message, duration=3000)
