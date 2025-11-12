# TaskMaster - Deployment Guide

## App Store Submission Checklist

### 1. Prepare Your App
- [ ] Test on physical device
- [ ] Fix all warnings and errors
- [ ] Optimize performance
- [ ] Add app icon (1024x1024)
- [ ] Add launch screen
- [ ] Configure privacy permissions

### 2. App Store Connect Setup
1. Create app in App Store Connect
2. Fill in app information
3. Upload screenshots (required sizes)
4. Write app description
5. Set pricing and availability

### 3. Archive & Upload
```bash
# In Xcode:
1. Product → Archive
2. Distribute App
3. Upload to App Store Connect
4. Submit for Review
```

### 4. TestFlight (Optional)
- Add internal testers
- Collect feedback
- Fix issues before public release

### 5. Post-Submission
- Monitor review status
- Respond to reviewer feedback
- Plan marketing strategy

## Monetization Setup

### In-App Purchases
1. Create products in App Store Connect
2. Update product IDs in Monetization.swift
3. Test with sandbox accounts

### Ads (AdMob)
1. Create AdMob account
2. Add app to AdMob
3. Update ad unit IDs

## Updates
- Increment version number
- Write release notes
- Submit update through same process

Generated: 2025-11-06
