{{/*
Expand the name of the chart.
*/}}
{{- define "omni-sense.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
Truncated at 63 chars because some Kubernetes name fields are limited.
*/}}
{{- define "omni-sense.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart label value: name-version.
*/}}
{{- define "omni-sense.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels applied to every resource.
*/}}
{{- define "omni-sense.labels" -}}
helm.sh/chart: {{ include "omni-sense.chart" . }}
app.kubernetes.io/part-of: omni-sense
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels for a specific component.
Usage: include "omni-sense.selectorLabels" (dict "component" "eep" "context" .)
*/}}
{{- define "omni-sense.selectorLabels" -}}
app.kubernetes.io/name: {{ .component }}
app.kubernetes.io/instance: {{ .context.Release.Name }}
{{- end }}

{{/*
Full component labels (common + selector).
Usage: include "omni-sense.componentLabels" (dict "component" "eep" "context" .)
*/}}
{{- define "omni-sense.componentLabels" -}}
{{ include "omni-sense.labels" .context }}
{{ include "omni-sense.selectorLabels" . }}
app.kubernetes.io/component: {{ .component }}
app.kubernetes.io/version: {{ .context.Chart.AppVersion | quote }}
{{- end }}

{{/*
Image reference helper.
Usage: include "omni-sense.image" (dict "repository" "eep" "context" .)
*/}}
{{- define "omni-sense.image" -}}
{{- printf "%s/%s:%s" .context.Values.image.registry .repository .context.Values.image.tag }}
{{- end }}

{{/*
Namespace helper — returns the configured namespace.
*/}}
{{- define "omni-sense.namespace" -}}
{{- .Values.namespace | default "omni-sense" }}
{{- end }}
