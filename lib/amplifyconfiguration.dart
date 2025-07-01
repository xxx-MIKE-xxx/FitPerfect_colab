const amplifyconfig = '''{
    "UserAgent": "aws-amplify-cli/2.0",
    "Version": "1.0",
    "auth": {
        "plugins": {
            "awsCognitoAuthPlugin": {
                "UserAgent": "aws-amplify-cli/0.1.0",
                "Version": "0.1.0",
                "IdentityManager": {
                    "Default": {}
                },
                "CredentialsProvider": {
                    "CognitoIdentity": {
                        "Default": {
                            "PoolId": "eu-central-1:2d3abfab-d5c6-4ab6-b2d7-47311a699e42",
                            "Region": "eu-central-1"
                        }
                    }
                },
                "CognitoUserPool": {
                    "Default": {
                        "PoolId": "eu-central-1_boU4KUcaB",
                        "AppClientId": "2jarpfv6l69pf3reotj11fcl2m",
                        "Region": "eu-central-1"
                    }
                },
                "Auth": {
                    "Default": {
                        "OAuth": {
                            "WebDomain": "fitperfect69642e95-69642e95-dev.auth.eu-central-1.amazoncognito.com",
                            "AppClientId": "2jarpfv6l69pf3reotj11fcl2m",
                            "SignInRedirectURI":  "com.fitperfect://auth",
                            "SignOutRedirectURI": "com.fitperfect://signout",
                            "Scopes": [
                                "email",
                                "openid",
                                "profile"
                            ]
                        },
                        "authenticationFlowType": "USER_SRP_AUTH",
                        "mfaConfiguration": "OFF",
                        "mfaTypes": [
                            "SMS"
                        ],
                        "passwordProtectionSettings": {
                            "passwordPolicyMinLength": 8,
                            "passwordPolicyCharacters": []
                        },
                        "signupAttributes": [
                            "EMAIL"
                        ],
                        "socialProviders": [
                            "FACEBOOK",
                            "GOOGLE"
                        ],
                        "usernameAttributes": [
                            "EMAIL"
                        ],
                        "verificationMechanisms": [
                            "EMAIL"
                        ]
                    }
                },
                "S3TransferUtility": {
                    "Default": {
                        "Bucket": "fitperfect-exercisevideos-deved5ec-dev",
                        "Region": "eu-central-1"
                    }
                }
            }
        }
    },
    "storage": {
        "plugins": {
            "awsS3StoragePlugin": {
                "bucket": "fitperfect-exercisevideos-deved5ec-dev",
                "region": "eu-central-1",
                "defaultAccessLevel": "guest"
            }
        }
    }
}''';
