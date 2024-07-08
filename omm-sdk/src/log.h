/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include <omm.h>

namespace omm
{
	class Logger
	{
	public:

		Logger() : m_log(ommMessageInterfaceDefault()) { }

		explicit Logger(ommMessageInterface log) :m_log(log) { }

		bool HasLogger() const
		{
			return m_log.messageCallback != nullptr;
		}

		void Info(const char* msg) const
		{
			if (m_log.messageCallback)
			{
				(*m_log.messageCallback)(ommMessageSeverity_Info, msg, m_log.userArg);
			}
		}

		void Warn(const char* msg) const
		{
			if (m_log.messageCallback)
			{
				(*m_log.messageCallback)(ommMessageSeverity_Warning, msg, m_log.userArg);
			}
		}

		void Fatal(const char* msg) const
		{
			if (m_log.messageCallback)
			{
				(*m_log.messageCallback)(ommMessageSeverity_Fatal, msg, m_log.userArg);
			}
		}

		// Helper versions that return specific error codes

		[[nodiscard]] ommResult InvalidArg(const char* msg) const
		{
			if (m_log.messageCallback)
			{
				(*m_log.messageCallback)(ommMessageSeverity_Fatal, msg, m_log.userArg);
			}
			return ommResult_INVALID_ARGUMENT;
		}

		template<int N = 256, typename... Args>
		[[nodiscard]] ommResult InvalidArgf(const char* format, Args&&... args) const
		{
			if (m_log.messageCallback)
			{
				char buffer[N];
				int result = sprintf_s(buffer, sizeof(buffer), format, std::forward<Args>(args)...);
				if (result < 0) {
					return ommResult_FAILURE; // sprintf_s failed for some reason
				}

				(*m_log.messageCallback)(ommMessageSeverity_Fatal, buffer, m_log.userArg);
			}
			return ommResult_INVALID_ARGUMENT;
		}

		[[nodiscard]] ommResult NotImplemented(const char* msg) const
		{
			if (m_log.messageCallback)
			{
				(*m_log.messageCallback)(ommMessageSeverity_Fatal, msg, m_log.userArg);
			}
			return ommResult_NOT_IMPLEMENTED;
		}

	private:
		ommMessageInterface m_log;
	};


} // namespace omm
