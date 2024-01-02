export $(cat .env)
export GOOGLE_APPLICATION_CREDENTIALS="../tform-intro-main.json"
terraform plan -var="gcp_project_id="$GOOGLE_PROJECT"" -var="service-account="$GOOGLE_ACCOUNT_NAME""